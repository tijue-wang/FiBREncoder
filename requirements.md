# Requirements

Target environment: Python 3.10, PyTorch 2.0.1, and torchvision 0.15.2.

## Base environment

```bash
conda create -n fibre python=3.10
conda activate fibre
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.4 scipy==1.10.1 scikit-image==0.21.0 opencv-python-headless==4.8.1.78 Pillow==10.4.0 matplotlib==3.7.5 h5py==3.10.0 einops==0.7.0 tqdm==4.66.5 albumentations==1.3.1
```

The PyTorch command targets a compatible NVIDIA GPU with CUDA 11.8. See the [official installation archive](https://pytorch.org/get-started/previous-versions/#v201) for other platforms.

## Nine-input Mamba network

Install these additional dependencies after PyTorch. The Mamba implementation requires Linux, a compatible NVIDIA GPU, and a compatible CUDA toolkit.

```bash
pip install timm==0.9.16 causal-conv1d==1.2.0.post2 mamba-ssm==1.2.0.post1 --no-build-isolation
```

These versions are a starting configuration, not a validated lockfile. Full training and inference with model weights have not yet been verified in this environment.
