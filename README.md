# Deep Quantum Error Correction

Implementation of the [Deep Quantum Error Correction paper (AAAI 2024)](https://arxiv.org/abs/2301.11930).

## Abstract

Quantum error correction codes (QECC) are a key component for realizing the potential of quantum computing. QECC, as its classical counterpart (ECC), enables the reduction of error rates, by distributing quantum logical information across redundant physical qubits, such that errors can be detected and corrected. In this work, we efficiently train novel deep quantum error decoders. We resolve the quantum measurement collapse by augmenting syndrome decoding to predict an initial estimate of the system noise, which is then refined iteratively through a deep neural network. The logical error rates calculated over finite fields are directly optimized via a differentiable objective, enabling efficient decoding under the constraints imposed by the code. Finally, our architecture is extended to support faulty syndrome measurement, to allow efficient decoding over repeated syndrome sampling. The proposed method demonstrates the power of neural decoders for QECC by achieving state-of-the-art accuracy, outperforming, for a broad range of topological codes, the existing neural and classical decoders, which are often computationally prohibitive.

## Install
- Pytorch
- Pymatching (for baselines results only)

## Script
Use the following command to train, on GPU 0, a 6 layers QECCT of dimension 128 on the L=4 Toric code, with the independent noise model, without repetitions (no faulty syndrome):

`python Main.py --gpus=0 --N_dec=6 --d_model=128 --code_L=4 --noise_type=independent --repetitions=1`

Use `--help` for additional arguments.

## Reference
    @article{choukroun2023deep,
      title={Deep Quantum Error Correction},
      author={Choukroun, Yoni and Wolf, Lior},
      journal={arXiv preprint arXiv:2301.11930},
      year={2023}
    }
