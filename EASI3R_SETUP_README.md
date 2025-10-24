# Easi3R Complete Setup Guide

Complete automated setup script for Easi3R with Pyenv and Miniforge, including all fixes for PyTorch 2.6 compatibility.

## Quick Start

```bash
# Download and run the setup script
chmod +x easi3r_complete_setup.sh
./easi3r_complete_setup.sh
```

By default, Easi3R will be installed in `~/Easi3R`. To install in a different location:

```bash
./easi3r_complete_setup.sh /path/to/installation/directory
```

## What the Script Does

The script performs a complete automated installation:

1. ✅ Checks prerequisites (pyenv, git)
2. ✅ Installs miniforge3-latest via pyenv
3. ✅ Configures conda channels (pytorch, nvidia, conda-forge)
4. ✅ Creates `easi3r` conda environment with Python 3.10
5. ✅ Installs PyTorch with CUDA support (default: CUDA 12.4)
6. ✅ Installs CUDA toolkit for compilation
7. ✅ Clones Easi3R repository
8. ✅ Installs all Python dependencies
9. ✅ Installs viser (4D visualization tool)
10. ✅ Installs SAM2
11. ✅ **Patches kernels.cu for PyTorch 2.6 compatibility**
12. ✅ Compiles RoPE CUDA kernels with proper CUDA_HOME

## Prerequisites

- **pyenv** installed and in PATH
  - Install: https://github.com/pyenv/pyenv#installation
- **git** installed
- **NVIDIA GPU** with compatible drivers (for CUDA support)
- **~5GB disk space** for the environment and dependencies

## CUDA Version Selection

The script will auto-detect your GPU and default to CUDA 12.4. You can specify a different version during installation:

- CUDA 12.4 (default)
- CUDA 12.1
- CUDA 11.8

## Key Features & Fixes

### 1. PyTorch 2.6 Compatibility Patch
The script automatically patches `kernels.cu` to fix the deprecated API:
```bash
# Changes tokens.type() to tokens.scalar_type()
sed -i 's/tokens\.type()/tokens.scalar_type()/g' kernels.cu
```

### 2. Proper CUDA_HOME Configuration
Sets `CUDA_HOME=$CONDA_PREFIX` before compilation to ensure nvcc is found.

### 3. Reliable PyTorch Installation
Uses pip with specific CUDA wheel URLs for consistent installation:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Error Handling
The script exits on any error (`set -e`) and provides clear error messages.

## Post-Installation Usage

After installation, activate your environment:

```bash
# Activate pyenv
pyenv global miniforge3-latest
eval "$(pyenv init -)"

# Activate conda environment
conda activate easi3r

# Navigate to Easi3R
cd ~/Easi3R  # or your custom installation path
```

### Make It Permanent

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Then you only need:
```bash
conda activate easi3r
```

## Verification

Check your installation:

```bash
conda activate easi3r

# Check Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check key packages
python -c "import sam2; print('SAM2: OK')"
python -c "import viser; print('viser: OK')"
```

## Troubleshooting

### Issue: "pyenv: command not found"
**Solution:** Install pyenv first:
```bash
curl https://pyenv.run | bash
```

### Issue: "conda: command not found" after installation
**Solution:** Source conda manually:
```bash
source ~/.pyenv/versions/miniforge3-latest/etc/profile.d/conda.sh
```

### Issue: CUDA not available in PyTorch
**Possible causes:**
1. No NVIDIA GPU in the system
2. NVIDIA drivers not installed
3. On WSL without proper GPU passthrough

**Solution:** 
- Check GPU: `nvidia-smi`
- Update drivers if needed
- For WSL: Ensure Windows has CUDA-capable drivers and WSL2 GPU support

### Issue: RoPE kernel compilation fails
**Impact:** Optional optimization only - Easi3R will still work
**Solution:** Skip it and continue. The kernels provide performance benefits but aren't required.

### Issue: "Permission denied" errors
**Solution:** Don't use `sudo` with conda/pip. Ensure pyenv directory has proper permissions.

## Manual Installation

If you prefer manual installation or need to customize steps, see the script source. Each step is clearly commented and can be run independently.

## Environment Management

### Updating Easi3R
```bash
conda activate easi3r
cd ~/Easi3R
git pull
pip install -r requirements.txt
```

### Removing the Environment
```bash
conda env remove -n easi3r
pyenv uninstall miniforge3-latest  # optional
rm -rf ~/Easi3R  # optional
```

### Creating a Backup
```bash
conda env export -n easi3r > easi3r_environment.yml
```

### Restoring from Backup
```bash
conda env create -f easi3r_environment.yml
```

## System Requirements

- **OS:** Linux (tested on Ubuntu 20.04+, WSL2)
- **Python:** 3.10 (installed by script)
- **CUDA:** 11.8+ (drivers must be pre-installed)
- **RAM:** 8GB minimum, 16GB+ recommended
- **GPU:** NVIDIA GPU with 6GB+ VRAM recommended

## For WSL Users

1. Install NVIDIA drivers on Windows (CUDA-capable)
2. Enable WSL2 GPU support
3. Run this script in WSL
4. CUDA toolkit will be installed via conda (don't install system CUDA in WSL)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify prerequisites are installed
3. Review the script output for specific error messages
4. Check Easi3R repository issues: https://github.com/Inception3D/Easi3R/issues

## License

This setup script is provided as-is for setting up Easi3R. Easi3R itself is subject to its own license terms.

## Credits

- Easi3R: https://github.com/Inception3D/Easi3R
- PyTorch: https://pytorch.org
- Miniforge: https://github.com/conda-forge/miniforge
- Pyenv: https://github.com/pyenv/pyenv
