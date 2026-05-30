# American Sign Language Recognition and Translation System

A comprehensive system for recognizing Vietnamese Sign Language (VSL) and translating it into Vietnamese and Lao languages using deep learning and NLP techniques.

## 🎯 Overview

This project combines three main components:

1. **VSL Recognition** - Real-time sign language recognition using MediaPipe and TensorFlow Lite
2. **Vietnamese NLP** - BART-PHO model for converting sign glosses to natural Vietnamese sentences
3. **Vi-Lao Translation** - NLLB model with LoRA fine-tuning for Vietnamese-Lao translation

## 📦 Project Structure

```
├── VSL_Vietnamese_NLP/          # NLP module
├── dich-vi_lo/                  # Translation module
├── external/                    # External baseline models
├── scripts/                     # Main processing scripts
├── Report_OP_LAB/               # Academic report (LaTeX)
├── Presentation_Demo/           # Presentation slides
├── docs/                        # Documentation
└── .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

```bash
# Clone and setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r VSL_Vietnamese_NLP/requirements.txt
pip install -r dich-vi_lo/requirements.txt
pip install gdown

# Download pre-trained models
python scripts/download_models.py

# Run integrated pipeline
python scripts/integrated_pipeline.py --input video.mp4 --output results.json

# Run real-time processing
python scripts/realtime_integrated.py --device 0
```

## 📚 Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) - Detailed system description
- [Project Structure](docs/PROJECT_STRUCTURE.md) - Directory organization
- [Setup Guide](docs/SETUP_GUIDE.md) - Detailed installation instructions
- [Integration Pipeline](docs/INTEGRATION_PIPELINE.md) - Pipeline architecture
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

## 📊 Key Features

- **Real-time Processing**: Optimized for webcam input
- **Multi-language Support**: Vietnamese and Lao outputs
- **End-to-End Pipeline**: From video to translated text
- **Modular Design**: Easy to integrate individual components
- **Well-documented**: Comprehensive setup and usage guides

## 🔬 Research Components

- Custom Vietnamese Sign Language dataset (250+ gesture classes)
- Fine-tuned BART-PHO for gloss-to-sentence conversion
- LoRA-adapted NLLB for Vi-Lao translation
- Detailed evaluation metrics and analysis

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{vsl_2026,
  title={Vietnamese Sign Language Recognition and Translation System},
  author={[Authors]},
  year={2026},
  url={[Repository URL]}
}
```



For detailed documentation, see the [docs/](docs/) directory.

