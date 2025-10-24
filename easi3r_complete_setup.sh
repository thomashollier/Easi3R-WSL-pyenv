#!/bin/bash
set -e  # Exit on any error

echo "============================================================"
echo "  Easi3R Complete Setup Script for Pyenv + Miniforge"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Set up miniforge3-latest in pyenv"
echo "  2. Create the easi3r conda environment"
echo "  3. Install all dependencies with CUDA support"
echo "  4. Clone Easi3R repository"
echo "  5. Install Python packages"
echo "  6. Patch and compile RoPE CUDA kernels"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# ==============================================================================
# Step 0: Prerequisites Check
# ==============================================================================
echo ""
echo "Step 0: Checking prerequisites..."

if ! command -v pyenv &> /dev/null; then
    echo "ERROR: pyenv is not installed or not in PATH"
    echo "Install pyenv first: https://github.com/pyenv/pyenv"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "ERROR: git is not installed"
    exit 1
fi

echo "✓ Prerequisites check passed"

# ==============================================================================
# Step 1: Install Miniforge via Pyenv
# ==============================================================================
echo ""
echo "Step 1: Installing miniforge3-latest via pyenv..."

if pyenv versions | grep -q "miniforge3-latest"; then
    echo "✓ miniforge3-latest already installed"
else
    echo "Installing miniforge3-latest (this may take a few minutes)..."
    pyenv install miniforge3-latest
fi

# Set pyenv to use miniforge
pyenv global miniforge3-latest
eval "$(pyenv init -)"

# Verify
if command -v conda &> /dev/null; then
    echo "✓ Conda is now available"
    conda --version
else
    # Try to source conda manually
    source ~/.pyenv/versions/miniforge3-latest/etc/profile.d/conda.sh
    if command -v conda &> /dev/null; then
        echo "✓ Conda is now available"
        conda --version
    else
        echo "ERROR: conda command not found after installation"
        exit 1
    fi
fi

# ==============================================================================
# Step 2: Configure Conda Channels
# ==============================================================================
echo ""
echo "Step 2: Configuring conda channels..."

conda config --add channels pytorch
conda config --add channels nvidia
conda config --set channel_priority flexible

echo "✓ Channels configured"

# ==============================================================================
# Step 3: Create Conda Environment
# ==============================================================================
echo ""
echo "Step 3: Creating easi3r conda environment..."

if conda env list | grep -q "^easi3r "; then
    echo "⚠ Environment 'easi3r' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n easi3r -y
        conda create -n easi3r python=3.10 cmake=3.31 -y
    else
        echo "Using existing environment"
    fi
else
    conda create -n easi3r python=3.10 cmake=3.31 -y
fi

echo "✓ Environment created"

# ==============================================================================
# Step 4: Activate Environment and Install PyTorch with CUDA
# ==============================================================================
echo ""
echo "Step 4: Activating environment and installing PyTorch with CUDA..."

# Activate the environment
source ~/.pyenv/versions/miniforge3-latest/etc/profile.d/conda.sh
conda activate easi3r

# Check for CUDA version (default to 12.4)
echo "Detecting CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    echo "Default: Installing PyTorch with CUDA 12.4 support"
    read -p "Press Enter to continue with CUDA 12.4, or type a different version (e.g., 11.8, 12.1): " CUDA_VERSION
    CUDA_VERSION=${CUDA_VERSION:-12.4}
else
    echo "⚠ nvidia-smi not found. Assuming CUDA 12.4"
    CUDA_VERSION=12.4
fi

# Install PyTorch with CUDA support using pip (more reliable)
echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
if [[ "$CUDA_VERSION" == "12.4" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch with conda for CUDA ${CUDA_VERSION}..."
    conda install pytorch torchvision pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y
fi

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✓ PyTorch with CUDA support installed successfully"
else
    echo "⚠ WARNING: PyTorch is installed but CUDA is not available"
    echo "The RoPE kernel compilation may fail, but Easi3R will still work (slower)"
fi

# ==============================================================================
# Step 5: Install CUDA Toolkit
# ==============================================================================
echo ""
echo "Step 5: Installing CUDA toolkit for compilation..."

conda install -c nvidia cuda-toolkit -y

echo "✓ CUDA toolkit installed"

# ==============================================================================
# Step 6: Clone Easi3R Repository
# ==============================================================================
echo ""
echo "Step 6: Cloning Easi3R repository..."

# Determine where to clone
INSTALL_DIR="${1:-$HOME/Easi3R}"
echo "Installation directory: $INSTALL_DIR"

if [ -d "$INSTALL_DIR" ]; then
    echo "⚠ Directory $INSTALL_DIR already exists"
    read -p "Skip cloning? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Using existing directory"
        cd "$INSTALL_DIR"
    else
        rm -rf "$INSTALL_DIR"
        git clone https://github.com/Inception3D/Easi3R.git "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
else
    git clone https://github.com/Inception3D/Easi3R.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo "✓ Repository ready at $INSTALL_DIR"

# ==============================================================================
# Step 7: Install Python Requirements
# ==============================================================================
echo ""
echo "Step 7: Installing Python requirements..."

pip install -r requirements.txt

echo "✓ Requirements installed"

# ==============================================================================
# Step 8: Install Viser (4D Visualization Tool)
# ==============================================================================
echo ""
echo "Step 8: Installing viser..."

pip install -e viser

echo "✓ Viser installed"

# ==============================================================================
# Step 9: Install SAM2
# ==============================================================================
echo ""
echo "Step 9: Installing SAM2..."

pip install -e third_party/sam2 --verbose

echo "✓ SAM2 installed"

# ==============================================================================
# Step 10: Compile RoPE CUDA Kernels
# ==============================================================================
echo ""
echo "Step 10: Compiling RoPE CUDA kernels..."

cd croco/models/curope/

# Set CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX
echo "CUDA_HOME set to: $CUDA_HOME"

# Backup original kernels.cu
if [ ! -f kernels.cu.backup ]; then
    echo "Creating backup of kernels.cu..."
    cp kernels.cu kernels.cu.backup
fi

# Apply PyTorch 2.6 compatibility patch
echo "Applying PyTorch 2.6 compatibility patch..."
sed -i 's/tokens\.type()/tokens.scalar_type()/g' kernels.cu

# Compile
echo "Compiling CUDA kernels (this may take a few minutes)..."
if python setup.py build_ext --inplace; then
    echo "✓ RoPE CUDA kernels compiled successfully"
else
    echo "⚠ WARNING: RoPE kernel compilation failed"
    echo "This is optional - Easi3R will still work but may be slower"
    echo "You can continue using Easi3R without these optimizations"
fi

# Go back to project root
cd ../../../

# ==============================================================================
# Step 11: Final Verification
# ==============================================================================
echo ""
echo "============================================================"
echo "  Final Verification"
echo "============================================================"

echo ""
echo "Python environment:"
python --version
echo ""

echo "PyTorch:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}')"
echo ""

echo "Key packages:"
python -c "import sam2; print('  SAM2: ✓')" 2>/dev/null || echo "  SAM2: ✗"
python -c "import viser; print('  viser: ✓')" 2>/dev/null || echo "  viser: ✗"
echo ""

# ==============================================================================
# Step 12: Setup Summary
# ==============================================================================
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "Installation directory: $INSTALL_DIR"
echo "Conda environment: easi3r"
echo ""
echo "To use Easi3R in the future:"
echo "  1. Activate pyenv and conda:"
echo "     pyenv global miniforge3-latest"
echo "     eval \"\$(pyenv init -)\"" 
echo "     conda activate easi3r"
echo ""
echo "  2. Navigate to Easi3R:"
echo "     cd $INSTALL_DIR"
echo ""
echo "  3. Run your Easi3R scripts"
echo ""
echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
echo "  export PYENV_ROOT=\"\$HOME/.pyenv\""
echo "  export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
echo "  eval \"\$(pyenv init -)\""
echo ""
echo "Happy 3D reconstructing! 🎉"
