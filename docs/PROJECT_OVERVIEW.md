# Project Overview - Vietnamese Sign Language Recognition and Translation System

## System Description

This project implements a comprehensive pipeline for recognizing Vietnamese Sign Language and translating it into both Vietnamese and Lao languages. The system combines computer vision, natural language processing, and machine translation techniques.

## System Architecture

### Component 1: VSL Recognition Module
- **Framework**: MediaPipe + TensorFlow Lite
- **Recognition Classes**: 250-class gesture recognition (extended from ASL)
- **Output**: Sequence of recognized signs (glosses)
- **Performance**: Real-time processing from webcam
- **Baseline**: Pre-trained ASL-250 model from Google

### Component 2: Vietnamese Sentence Reconstruction
- **Model**: BART-PHO (PhoBERT-based) fine-tuned on Vietnamese
- **Task**: Gloss-to-sentence conversion
- **Components**:
  - Vocabulary mapping (gloss to Vietnamese words)
  - Sentence rules engine for linguistic constraints
  - Fluency optimization
- **Output**: Natural Vietnamese sentences from sign sequences

### Component 3: Vietnamese to Lao Translation
- **Base Model**: NLLB (No Language Left Behind) by Meta
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficiency
- **Features**:
  - Bidirectional translation (Vietnamese ↔ Lao)
  - Optimized for low-resource language pairs
  - Fast inference with quantization support

### Component 4: Real-time Integration Pipeline
- **Input**: Video stream (file or webcam)
- **Processing**: 
  - Frame-by-frame sign recognition
  - Temporal sliding window for context
  - Gloss sequence aggregation
  - Sentence generation
  - Translation to Lao
- **Output**: Multilingual text (Vietnamese, Lao)

## Project Structure

```
project-root/
│
├── VSL_Vietnamese_NLP/                    # NLP Module
│   ├── configs/
│   │   ├── nlp_config.json               # NLP configuration
│   │   └── sentence_rules.json           # Linguistic rules
│   ├── data/
│   │   ├── gloss_vocabulary_full_v4.csv  # Gloss-to-word mapping
│   │   ├── sentence_pairs*.csv           # Training data
│   │   └── README_DATASET.md             # Dataset documentation
│   ├── models/
│   │   └── bartpho_sentence/             # Fine-tuned BART model
│   ├── src/
│   │   ├── inference/                    # Prediction pipelines
│   │   ├── nlp/                          # NLP utilities
│   │   └── training/                     # Training scripts
│   ├── tests/
│   └── requirements.txt
│
├── dich-vi_lo/                            # Translation Module
│   ├── dataset/
│   │   └── vi_lo_dataset_split/          # Train/Valid/Test splits
│   ├── model/
│   │   └── vi_to_lao_nllb_lora/          # LoRA-adapted model
│   ├── translate.py                      # Translation interface
│   └── requirements.txt
│
├── external/                              # External Components
│   └── google-asl-250/
│       ├── model.tflite                  # ASL recognition model
│       ├── realtime_asl_250.py           # Real-time inference
│       ├── evaluate_asl_250.py           # Evaluation script
│       └── sign_to_prediction_index_map.json
│
├── scripts/                               # Main Scripts
│   ├── integrated_pipeline.py            # Complete end-to-end pipeline
│   └── realtime_integrated.py            # Real-time webcam processing
│
├── Report_OP_LAB/                        # Academic Report (LaTeX)
│   ├── chapters/                         # Report sections
│   ├── figures/                          # Diagrams and visualizations
│   ├── backmatter/
│   │   └── references.tex                # Bibliography
│   └── main.pdf                          # Compiled report
│
├── Presentation_Demo/                    # Presentation Slides (Beamer)
│   └── main.pdf                          # Compiled slides
│
├── docs/                                 # Documentation
│   ├── PROJECT_OVERVIEW.md              # This file
│   ├── PROJECT_STRUCTURE.md             # Directory organization
│   ├── SETUP_GUIDE.md                   # Installation & setup
│   ├── INTEGRATION_PIPELINE.md          # Pipeline details
│   ├── EXPERT_REVIEW.md                 # Review notes
│   └── MODEL_BASELINE.md                # Baseline model info
│
├── README.md                             # Main README
├── CONTRIBUTING.md                       # Contribution guidelines
└── .gitignore                            # Git ignore rules
```

## Installation & Usage

### Prerequisites
- Python 3.8+
- pip/conda
- OpenCV (for video processing)
- CUDA/cuDNN (optional, for GPU acceleration)

### Installation

1. **Clone repository and setup environment**
   ```bash
   git clone [repository-url]
   cd [project-directory]
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   cd VSL_Vietnamese_NLP
   pip install -r requirements.txt
   cd ../dich-vi_lo
   pip install -r requirements.txt
   cd ..
   ```

3. **Download pre-trained models** (if not included)
   - Models should be in respective module directories
   - See individual module READMEs for details

### Usage Examples

**Process video file:**
```bash
python scripts/integrated_pipeline.py \
  --input path/to/video.mp4 \
  --output results.json \
  --include-lao
```

**Real-time webcam:**
```bash
python scripts/realtime_integrated.py \
  --device 0 \
  --show-glosses \
  --show-vietnamese \
  --show-lao
```

**Individual modules:**

Vietnamese reconstruction:
```bash
cd VSL_Vietnamese_NLP
python -m src.inference.predict --glosses "sign1 sign2 sign3"
```

Vi-Lao translation:
```bash
cd dich-vi_lo
python translate.py --text "Hello world" --direction vi-lo
```

## Datasets

### Vietnamese Sign Language Dataset
- **Classes**: 250+ gesture types
- **Samples**: Custom collection from native signers
- **Format**: Video frames + keypoint annotations
- **Splits**: Train/Validation/Test

### Vietnamese-Lao Parallel Corpus
- **Size**: ~50K sentence pairs (varies by version)
- **Domain**: General conversation
- **Splits**: Train/Validation/Test
- **Location**: `dich-vi_lo/dataset/vi_lo_dataset_split/`

### Sentence Pairs Dataset
- **Format**: CSV (gloss_sequence, vietnamese_sentence)
- **Location**: `VSL_Vietnamese_NLP/data/`
- **Usage**: Training BART-PHO model

See [VSL_Vietnamese_NLP/data/README_DATASET.md](../VSL_Vietnamese_NLP/data/README_DATASET.md) for detailed dataset information.

## Evaluation Metrics

### Recognition Evaluation
- Accuracy, Precision, Recall, F1-Score
- Per-class performance analysis
- Confusion matrices

### NLP Evaluation
- BLEU, ROUGE, METEOR scores
- Sentence length statistics
- Vocabulary coverage

### Translation Evaluation
- TER (Translation Error Rate)
- chrF (character n-gram F-score)
- BLEU on test set
- Human evaluation results

Detailed results in [Report_OP_LAB/](../Report_OP_LAB/)

## Key References

Research papers and resources used:
- MediaPipe: Real-time pose estimation
- BART-PHO: Vietnamese BART model
- NLLB: Machine translation for 200+ languages
- LoRA: Efficient fine-tuning technique

See [Report_OP_LAB/backmatter/references.tex](../Report_OP_LAB/backmatter/references.tex) for full bibliography.

## Research Contributions

This work demonstrates:
- Practical Vietnamese Sign Language recognition system
- Effective gloss-to-sentence generation
- LoRA-based efficient translation model
- End-to-end integration of multiple NLP components
- Real-time processing pipeline

## Performance Characteristics

- **Recognition Latency**: ~50-100ms per frame (GPU)
- **Sentence Generation**: ~200-500ms per gloss sequence
- **Translation**: ~100-200ms per sentence
- **Total Pipeline**: <2s for typical sentence (GPU)

## Limitations & Future Work

### Current Limitations
- Limited to frontal sign configurations
- Model coverage depends on training data
- Translation quality varies by domain
- Requires GPU for optimal performance

### Future Improvements
- Extended gesture library
- Multi-angle recognition support
- Domain-specific translation models
- Mobile deployment optimization
- Real-time streaming improvements

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

[Specify your license]

## Acknowledgments

- MediaPipe team for pose estimation framework
- Hugging Face for transformer models
- Native signers who contributed data
- PTIT for research support

## Citation

```bibtex
@software{vsl_system_2026,
  title={Vietnamese Sign Language Recognition and Translation System},
  author={[Authors]},
  year={2026},
  url={[Repository URL]},
  note={Report available in Report_OP_LAB/main.pdf}
}
```

---

For specific component details, see the module README files:
- [VSL_Vietnamese_NLP/README.md](../VSL_Vietnamese_NLP/README.md)
- [dich-vi_lo/README.md](../dich-vi_lo/README.md)
- [external/asl_baseline/README.md](../external/asl_baseline/README.md)

