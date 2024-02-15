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
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

##################################################################
class QECC_Dataset(data.Dataset):
    def __init__(self, code, ps, len, args):
        self.code = code
        self.ps = ps
        self.len = len
        self.logic_matrix = code.logic_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1).clone().cpu()
        self.zero_cw = torch.zeros((self.pc_matrix.shape[0])).long()
        self.noise_method = self.independent_noise if args.noise_type == 'independent' else self.depolarization_noise
        self.args = args
        
    def independent_noise(self,pp=None):
        pp = random.choice(self.ps) if pp is None else pp
        return np.random.binomial(1, pp, self.pc_matrix.shape[0])
    
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
            syndrome = bin_to_sign(syndrome) 
            return x.float(), z.float(), y.float(), (magnitude*0+1).float(), syndrome.float()
        ###
        ### See original setting definition in https://pymatching.readthedocs.io/en/stable/toric-code-example.html# 
        qq = pp
        
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
        syndrome = bin_to_sign(torch.from_numpy(noisy_syndrome)) #TODO: check if bin2sign is needed

        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        return x.float(), z.float(), (y*0+1).float(), (magnitude*0+1).float(), syndrome.float().transpose(0,1)
    
    def __len__(self):
        return self.len


##################################################################
##################################################################
class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        return grad_output*(torch.abs(x[0])<=1)

def binarization(y):
    return sign_to_bin(Binarization.apply(y))
###
def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2
###
def diff_GF2_mul(H,x):    
    H_bin = sign_to_bin(H) if -1 in H else H
    x_bin = x
    
    tmp = bin_to_sign(H_bin.unsqueeze(0)*x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp,1)
    tmp = sign_to_bin(tmp)

    # assert torch.allclose(logical_flipped(H_bin,x_bin).cpu().detach().bool(), tmp.detach().cpu().bool())
    return tmp
##################################################################

def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ber = cum_ler = cum_samples = 0
    cum_loss1 = cum_loss2 = cum_loss3 = 0
    t = time.time()
    # bin_fun = binarization
    bin_fun = torch.sigmoid
    for batch_idx, (x, z, y, magnitude, syndrome) in enumerate(train_loader):
        syndrome = syndrome.to(device)
        z_pred = model(magnitude.to(device), syndrome)
        loss1,loss2 = model.module.loss(-z_pred, z.to(device))
        loss3 = torch.nn.functional.binary_cross_entropy_with_logits((diff_GF2_mul(train_loader.dataset.logic_matrix,bin_fun(-z_pred))),logical_flipped(train_loader.dataset.logic_matrix, z.to(device)))
        loss = args.lambda_loss_ber*loss1+args.lambda_loss_n_pred*loss2+args.lambda_loss_ler*loss3
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        z_pred = sign_to_bin(torch.sign(-z_pred))
        ber = BER(z_pred, z.to(device))
        ler = FER(logical_flipped(train_loader.dataset.logic_matrix, z_pred), logical_flipped(train_loader.dataset.logic_matrix, z.to(device)) )

        cum_loss += loss.item() * z.shape[0]
        #
        cum_loss1 += loss1.item() * z.shape[0]
        cum_loss2 += loss2.item() * z.shape[0]
        cum_loss3 += loss3.item() * z.shape[0]
        #
        cum_ber += ber * z.shape[0]
        cum_ler += ler * z.shape[0]
        cum_samples += z.shape[0]
        #
        if (batch_idx+1) % (len(train_loader)//2) == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e} BER={cum_ber / cum_samples:.3e} LER={cum_ler / cum_samples:.3e}')
            logging.info(
                f'***Loss={cum_loss / cum_samples:.5e} Loss LER={cum_loss3 / cum_samples:.5e} Loss BER={cum_loss1 / cum_samples:.5e} Loss noise pred={cum_loss2 / cum_samples:.5e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_ler / cum_samples


##################################################################

def test(model, device, test_loader_list, ps_range_test, cum_count_lim=100000):
    model.eval()
    test_loss_ber_list, test_loss_ler_list, cum_samples_all = [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_ler = cum_count = 0.
            while True:
                (x, z, y, magnitude, syndrome) = next(iter(test_loader))
                z_pred = model(magnitude.to(device), syndrome.to(device))
                _ = model.module.loss(-z_pred, z.to(device))
                z_pred = sign_to_bin(torch.sign(-z_pred))

                test_ber += BER(z_pred, z.to(device)) * z.shape[0]
                test_ler += FER(logical_flipped(test_loader.dataset.logic_matrix, z_pred), logical_flipped(test_loader.dataset.logic_matrix, z.to(device)) ) * z.shape[0]
                cum_count += z.shape[0]
                if cum_count > cum_count_lim:
                    break
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_ler_list.append(test_ler / cum_count)
            print(f'Test p={ps_range_test[ii]:.3e}, BER={test_loss_ber_list[-1]:.3e}, LER={test_loss_ler_list[-1]:.3e}')
        ###
        logging.info('Test LER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ler_list, ps_range_test))]))
        logging.info('Test BER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, ps_range_test))]))
        logging.info(f'Mean LER = {np.mean(test_loss_ler_list):.3e}, Mean BER = {np.mean(test_loss_ber_list):.3e}')
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ber_list, test_loss_ler_list

##################################################################
##################################################################
##################################################################


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.code.logic_matrix = args.code.logic_matrix.to(device) 
    args.code.pc_matrix = args.code.pc_matrix.to(device) 
    code = args.code
    assert 0 < args.repetitions 
    if args.repetitions > 1:
        from Model_T_measurements import ECC_Transformer
    else:
        from Model import ECC_Transformer

    #################################
    model = ECC_Transformer(args, dropout=0).to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    logging.info(f'PC matrix shape {code.pc_matrix.shape}')
    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    ps_test = np.linspace(0.01, 0.2, 9)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, 0.2, 9)
    if args.repetitions > 1:
        ps_test = np.linspace(0.02, 0.04, 9)
    ###
    ps_train = ps_test

    train_dataloader = DataLoader(QECC_Dataset(code, ps_train, len=args.batch_size * 5000, args=args), batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(QECC_Dataset(code, [ps_test[ii]], len=int(args.test_batch_size),args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(ps_test))]
    #################################
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ber, ler = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()
        torch.save(model, os.path.join(args.path, 'last_model'))
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
            logging.info('Model Saved')
        if epoch % 60 == 0 or epoch in [1, args.epochs]:
            test(model, device, test_dataloader_list, ps_test)
    ###
    model = torch.load(os.path.join(args.path, 'best_model')).to(device)
    logging.info('Best model loaded')
    ps_test = np.linspace(0.01, 0.2, 18)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, 0.2, 18)
    if args.repetitions > 1:
        ps_test = np.linspace(0.02, 0.04, 18)
    ###
    test_dataloader_list = [DataLoader(QECC_Dataset(code, [ps_test[ii]], len=int(args.test_batch_size),args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(ps_test))]
    test(model, device, test_dataloader_list, ps_test)
##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DQEC')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpus', type=str, default='0', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='toric',choices=['toric'])
    parser.add_argument('--code_L', type=int, default=4,help='Lattice length')
    parser.add_argument('--repetitions', type=int, default=1,help='Number of faulty repetitions. <=1 is equivalent to none.')
    parser.add_argument('--noise_type', type=str,default='independent', choices=['independent','depolarization'],help='Noise model')

    # model args
    parser.add_argument('--N_dec', type=int, default=6,help='Number of QECCT self-attention modules')
    parser.add_argument('--d_model', type=int, default=128,help='QECCT dimension')
    parser.add_argument('--h', type=int, default=16,help='Number of heads')

    # qecc args
    parser.add_argument('--lambda_loss_ber', type=float, default=0.5,help='BER loss regularization')
    parser.add_argument('--lambda_loss_ler', type=float, default=1.,help='LER loss regularization')
    parser.add_argument('--lambda_loss_n_pred', type=float, default=0.5,help='g noise prediction regularization')
    
    # ablation args
    parser.add_argument('--no_g', type=int, default=0)
    parser.add_argument('--no_mask', type=int, default=0)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################
    if args.no_g > 0:
        args.lambda_loss_n_pred= 0.
    ###
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
    model_dir = os.path.join('Final_Results_QECCT', args.code_type, 
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

