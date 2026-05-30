# Setup and Installation Guide

Complete guide for setting up the Vietnamese Sign Language Recognition and Translation System.

## Prerequisites

### System Requirements
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Storage**: 20 GB free space (includes models)
- **GPU** (optional): NVIDIA GPU with CUDA support for faster processing

### Python Installation
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify pip is installed
pip --version
```

## Step 1: Clone Repository

```bash
# Clone the repository
git clone [repository-url]
cd [project-directory]

# List contents to verify structure
ls -la
# Should show: VSL_Vietnamese_NLP, dich-vi_lo, external, scripts, docs, etc.
```

## Step 2: Create Virtual Environment

Creating a virtual environment isolates project dependencies:

### On Windows
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (you should see (venv) in command prompt)
```

### On macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in prompt)
```

## Step 3: Upgrade pip and Install Build Tools

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install build tools
pip install setuptools wheel
```

## Step 4: Install Core Dependencies

```bash
# Install main requirements (from project root)
pip install numpy opencv-python mediapipe tensorflow
```

## Step 5: Install Module Dependencies

### Vietnamese NLP Module

```bash
cd VSL_Vietnamese_NLP
pip install -r requirements.txt

# Expected packages:
# - transformers (BART-PHO model)
# - torch (PyTorch)
# - scikit-learn (for evaluation)
# - pandas (for data processing)

cd ..
```

### Translation Module

```bash
cd dich-vi_lo
pip install -r requirements.txt

# Expected packages:
# - transformers (NLLB model)
# - peft (for LoRA)
# - torch (PyTorch)
# - safetensors (for model format)

cd ..
```

## Step 6: Verify Installation

### Test 1: Check Python Imports

```bash
python -c "
import cv2
import mediapipe
import tensorflow
import torch
import transformers
print('✓ All main packages installed successfully')
"
```

### Test 2: Check Module Imports

```bash
# Test VSL_Vietnamese_NLP imports
python -c "
import sys
sys.path.insert(0, 'VSL_Vietnamese_NLP')
from src.nlp.sentence_builder import SentenceBuilder
print('✓ VSL_Vietnamese_NLP imports working')
"

# Test dich-vi_lo imports
python -c "
import sys
sys.path.insert(0, 'dich-vi_lo')
from translate import VietnameseToLaoTranslator
print('✓ dich-vi_lo imports working')
"
```

## Step 7: Download Pre-trained Models

### Automatic Download

Models download automatically on first use, but you can pre-download them:

```bash
# Download BART-PHO for Vietnamese NLP
cd VSL_Vietnamese_NLP
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = 'VietAI/bartpho-syllable'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print('✓ BART-PHO model downloaded')
"
cd ..

# Download NLLB for translation
cd dich-vi_lo
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = 'facebook/nllb-200-distilled-600M'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print('✓ NLLB model downloaded')
"
cd ..
```

### Manual Download

Store models in their respective directories:
```
VSL_Vietnamese_NLP/models/
└── bartpho_sentence/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── ...

dich-vi_lo/model/
└── vi_to_lao_nllb_lora/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

## Step 8: Configure GPU Support (Optional)

### NVIDIA GPU with CUDA

```bash
# Uninstall CPU-only PyTorch
pip uninstall torch -y

# Install CUDA-enabled PyTorch (check PyTorch website for latest command)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
```

### macOS with Metal

```bash
# PyTorch Metal acceleration (built-in for newer versions)
python -c "
import torch
print('Metal acceleration available:', torch.backends.mps.is_available())
"
```

## Step 9: Test Basic Functionality

### Test 1: Simple Gesture Recognition

```bash
# Test MediaPipe setup
python -c "
import mediapipe as mp
print('✓ MediaPipe initialized')
print('Version:', mp.__version__)
"
```

### Test 2: Run Integrated Pipeline

```bash
# Test without translation (faster)
python scripts/integrated_pipeline.py --no-translation

# When prompted, enter sample input:
# hello
# /build
```

Expected output:
```
Token sequence: xin_chao
Vietnamese: Xin chào.
```

### Test 3: Test Real-time Processing

```bash
# Test webcam access
python scripts/realtime_integrated.py --no-translation --dry-run
```

Should initialize camera and display frame preview.

## Troubleshooting Installation

### Issue: pip command not found
```bash
# Try using python -m pip instead
python -m pip install --upgrade pip
python -m pip install -r VSL_Vietnamese_NLP/requirements.txt
```

### Issue: Permission denied (Linux/macOS)
```bash
# Use --user flag
pip install --user package_name

# Or use venv (recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "No module named 'torch'"
```bash
# Reinstall PyTorch
pip uninstall torch -y
pip install torch

# Or for GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "CUDA out of memory"
```python
# Reduce batch size in config files
# Or use CPU with:
# CUDA_VISIBLE_DEVICES="" python scripts/realtime_integrated.py
```

### Issue: Models not downloading
```bash
# Set HF_HOME to custom location with more space
export HF_HOME=/path/to/large/disk/huggingface

# Then run pipeline (will download there)
python scripts/integrated_pipeline.py
```

## Environment Configuration

### Create .env file (optional)
```bash
# Create .env file in project root
cat > .env << EOF
# Model paths
MODELS_PATH=./models
HF_HOME=./models/huggingface

# Processing
DEVICE=cuda  # or 'cpu'
NUM_WORKERS=4

# Logging
LOG_LEVEL=INFO
EOF
```

## Verify Complete Setup

```bash
# Run comprehensive check
python -c "
import sys
print('Python version:', sys.version)
print()

# Check all packages
packages = ['cv2', 'mediapipe', 'tensorflow', 'torch', 'transformers', 'peft']
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f'✓ {pkg}: {mod.__version__}')
    except ImportError:
        print(f'✗ {pkg}: NOT INSTALLED')

print()
print('✓ Setup verification complete!')
"
```

## System Information

To report issues, gather system info:

```bash
python -c "
import platform
import torch
import tensorflow as tf

print('System:', platform.system(), platform.release())
print('Python:', platform.python_version())
print('PyTorch:', torch.__version__)
print('TensorFlow:', tf.__version__)
print('GPU available (PyTorch):', torch.cuda.is_available())
print('GPU available (TensorFlow):', tf.test.is_built_with_cuda())
"
```

## Next Steps

1. **Quick Start**: Follow [README.md](../README.md) for basic usage
2. **Detailed Examples**: See [INTEGRATION_PIPELINE.md](./INTEGRATION_PIPELINE.md)
3. **Module Documentation**: 
   - [VSL_Vietnamese_NLP/README.md](../VSL_Vietnamese_NLP/README.md)
   - [dich-vi_lo/README.md](../dich-vi_lo/README.md)
4. **Development**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Getting Help

### Common Issues Resources
- [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) - System details
- [INTEGRATION_PIPELINE.md](./INTEGRATION_PIPELINE.md) - Pipeline troubleshooting
- Module READMEs for component-specific help

### Support Channels
- GitHub Issues: [Create an issue]
- Documentation: Check docs/ folder
- Email: [Support contact]

## Advanced Setup

### For Development

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### For GPU Development

```bash
# NVIDIA Container Toolkit (Docker)
# Install Docker: https://docs.docker.com/get-docker/
# Install NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-docker

# Run in container (optional)
docker build -t vsl-system .
docker run --gpus all -it vsl-system
```

### For Large-Scale Processing

```bash
# Install distributed processing tools
pip install ray joblib dask

# Configure for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust for your GPUs
```

---

**Setup complete!** Proceed to [README.md](../README.md) for usage instructions.

