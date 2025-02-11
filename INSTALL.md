
### 1. Basic Setup

```
# Create a new conda environment named 'lumina_video' with Python 3.11
conda create -n lumina_video python=3.11 -y
# Activate the 'lumina_video' environment
conda activate lumina_video
# Install required packages from 'requirements.txt'
pip install -r requirements.txt
```

### 2. Install Flash-Attention
```
pip install flash-attn --no-build-isolation
```
For reference, we use version 2.7.2.post1 in our environment.

### 3. Optional: Install Apex
> [!Caution]
>
> For both training and inference, Apex may bring some efficiency improvement, but it is not a must.
>
> Note that the code works smoothly with either:
> 1. Apex not installed at all; OR
> 2. Apex successfully installed with CUDA and C++ extensions.
>
> However, it will fail when:
> 1. A Python-only build of Apex is installed.
>
> If errors like `No module named 'fused_layer_norm_cuda'` are reported, it generally means that you are
using a Python-only Apex build. Please run `pip uninstall apex` to remove the build and try again.

Lumina-Video utilizes [apex](https://github.com/NVIDIA/apex), which needs to be compiled from source, for improved efficiency.
Please follow the [official instructions](https://github.com/NVIDIA/apex#from-source) for installation.
Here are some tips based on our experiences:

**Step1**: Check the version of CUDA with which your torch is built:
 ```python
# python
import torch
print(torch.version.cuda)
```

**Step2**: Check the CUDA toolkit version on your system:
```bash
# bash
nvcc -V
```
**Step3**: If the two aforementioned versions mismatch, or if you do not have CUDA toolkit installed on your system,
please download and install CUDA toolkit from [here](https://developer.nvidia.com/cuda-toolkit-archive) with version matching the torch CUDA version.

> [!Note]
>
> Note that multiple versions of CUDA toolkit can co-exist on the same machine, and the version can be easily switched by changing the `$PATH` and `$LD_LIBRARY_PATH` environment variables.
There is thus no need to worry about your machine's environment getting messed up.

**Step4**: You can now start installing apex:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
