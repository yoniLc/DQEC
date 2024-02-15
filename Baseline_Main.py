"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
import time
from pymatching import Matching

##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

##################################################################
def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2

class ECC_Dataset(data.Dataset):
    def __init__(self, code, ps, len, args):
        self.code = code
        self.ps = ps
        self.len = len
        self.logic_matrix = code.logic_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)
        self.zero_cw = torch.zeros((self.pc_matrix.shape[0])).long()
        self.noise_method = self.independent_noise if args.noise_type == 'independent' else self.depolarization_noise
        self.args = args
        
    def independent_noise(self,pp=None):
        pp = random.choice(self.ps) if pp is None else pp
        flips = np.random.binomial(1, pp, self.pc_matrix.shape[0])
        while not np.any(flips):
            flips = np.random.binomial(1, pp, self.pc_matrix.shape[0])
        return flips
    
    def depolarization_noise(self,pp=None):
        ## See original noise definition in https://github.com/Krastanov/neural-decoder/
        pp = random.choice(self.ps) if pp is None else pp
        out_dimZ = out_dimX = self.pc_matrix.shape[0]//2
        def makeflips(q):
            q = q/3.
            flips = np.zeros((out_dimZ+out_dimX,), dtype=np.dtype('b'))
            rand = np.random.rand(out_dimZ or out_dimX)
            both_flips  = (2*q<=rand) & (rand<3*q)
            ###
            x_flips = rand<  q
            flips[:out_dimZ] ^= x_flips
            flips[:out_dimZ] ^= both_flips
            ###
            z_flips = (q<=rand) & (rand<2*q)
            flips[out_dimZ:out_dimZ+out_dimX] ^= z_flips
            flips[out_dimZ:out_dimZ+out_dimX] ^= both_flips
            return flips
        flips = makeflips(pp)
        while not np.any(flips):
            flips = makeflips(pp)
        return flips*1.
        
        
    
    def __getitem__(self, index):
        x = self.zero_cw
        pp = random.choice(self.ps)
        if self.args.repetitions <= 1:
            z = torch.from_numpy(self.noise_method(pp))
            y = bin_to_sign(x) + z
            magnitude = torch.abs(y)
            syndrome = torch.matmul(z.long(),
                                    self.pc_matrix) % 2
            syndrome = syndrome
            return x.float(), z.float(), y.float(), (magnitude*0+1).float(), syndrome.float()
        ###
        qq = pp
        ### See original setting definition in https://pymatching.readthedocs.io/en/stable/toric-code-example.html# 

        noise_new = np.stack([self.noise_method(pp) for _ in range(self.args.repetitions)],1)
        noise_cumulative = (np.cumsum(noise_new, 1) % 2).astype(np.uint8)
        noise_total = noise_cumulative[:,-1]
        syndrome = (torch.matmul(torch.from_numpy(noise_cumulative).long().transpose(0,1),self.pc_matrix) % 2).transpose(0,1).numpy()
        syndrome_error = (np.random.rand(self.pc_matrix.shape[1], self.args.repetitions) < qq).astype(np.uint8)
        syndrome_error[:,-1] = 0 # Perfect measurements in last round to ensure even parity
        noisy_syndrome = (syndrome + syndrome_error) % 2
        # Convert to difference syndrome
        noisy_syndrome[:,1:] = (noisy_syndrome[:,1:] - noisy_syndrome[:,0:-1]) % 2
        
        z = torch.from_numpy(noise_total)
        syndrome = torch.from_numpy(noisy_syndrome)

        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        return x.float(), z.float(), (y*0+1).float(), (magnitude*0+1).float(), syndrome.float()
    
    def __len__(self):
        return self.len


##################################################################
def test(args,model, device, test_loader_list, ps_range_test, cum_count_lim=100000):
    test_loss_ber_list, test_loss_ler_list, cum_samples_all = [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_ler = cum_count = 0.
            while True:
                (x, z, y, magnitude, syndrome) = next(iter(test_loader))
                z_pred = []
                for ssynd in syndrome:
                    ssynd = ssynd.numpy()
                    if args.decoder == 'u-f':
                        ssynd = ssynd.T
                    z_pred.append(torch.from_numpy(model(ssynd).astype(float))) # Transpose is for U-F
                z_pred = torch.stack(z_pred).to(device)
                
                test_ber += BER(z_pred, z.to(device)) * z.shape[0]
                test_ler += FER(logical_flipped(test_loader.dataset.logic_matrix, z_pred), logical_flipped(test_loader.dataset.logic_matrix, z.to(device)) ) * z.shape[0]
                cum_count += z.shape[0]
                if cum_count > cum_count_lim:
                    break
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_ler_list.append(test_ler / cum_count)
            print(f'Test p={ps_range_test[ii]:.2e}, BER={test_loss_ber_list[-1]:.2e}, LER={test_loss_ler_list[-1]:.2e}')
        ###
        logging.info('Test LER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ler_list, ps_range_test))]))
        logging.info('Test BER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, ps_range_test))]))
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ber_list, test_loss_ler_list

##################################################################
##################################################################
##################################################################


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.code.logic_matrix = args.code.logic_matrix.to(device) 
    code = args.code
    assert 0 < args.repetitions 
    #################################
    ps_test = np.linspace(0.01, 0.2, 18)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, 0.2, 18)
    if args.repetitions > 1:
        ps_test = np.linspace(0.02, 0.04, 18)
    ####
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [ps_test[ii]], len=int(args.test_batch_size),args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(ps_test))]
    if args.decoder == 'u-f':
        from scipy import sparse
        from UnionFindPy import Decoder
        model = Decoder(sparse.csr_matrix(code.pc_matrix.int().numpy()),repetitions=args.repetitions if args.repetitions > 1 else None)
    else:
        model = Matching.from_check_matrix(code.pc_matrix,repetitions=args.repetitions if args.repetitions > 1 else None)
    logging.info(f'Model = {model}')
    logging.info(f'PC matrix shape {code.pc_matrix.shape}')
    model = model.decode
    test(args,model, device, test_dataloader_list, ps_test)
##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch QECCT')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='toric',choices=['toric','surface'])
    parser.add_argument('--code_L', type=int, default=8)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--noise_type', type=str,default='depolarization', choices=['independent','depolarization'])
    #    
    parser.add_argument('--decoder', type=str,default='mwpm', choices=['u-f','mwpm'])

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    class Code():
        pass
    code = Code()
    H,Lx = eval(f'Get_{args.code_type}_Code')(args.code_L,full_H=args.noise_type == 'depolarization')
    code.logic_matrix = torch.from_numpy(Lx).long()
    code.pc_matrix = torch.from_numpy(H).long()
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = args.code_type
    args.code = code
    ####################################################################
    model_dir = os.path.join('Baseline_Results_QDEC', args.decoder,args.code_type, 
                             'Code_L_' + str(args.code_L) , 
                             f'noise_model_{args.noise_type}', 
                             f'repetition_{args.repetitions}' , 
                             datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    main(args)
