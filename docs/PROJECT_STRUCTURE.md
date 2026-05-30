# Project Structure & Guidelines

## Directory Organization

```
project-root/
│
├── VSL_Vietnamese_NLP/              # Main NLP module
│   ├── configs/                     # Configuration files
│   ├── data/                        # Datasets and vocabulary
│   ├── models/                      # Pre-trained models
│   ├── src/                         # Source code
│   │   ├── inference/               # Inference pipelines
│   │   ├── nlp/                     # NLP utilities
│   │   └── training/                # Training scripts
│   ├── tests/                       # Unit tests
│   ├── main.py                      # Entry point
│   ├── requirements.txt             # Dependencies
│   └── README.md                    # Module-specific README
│
├── dich-vi_lo/                      # Translation module (Vi-Lao)
│   ├── dataset/                     # Training data splits
│   ├── model/                       # Translation model (LoRA)
│   ├── translate.py                 # Translation script
│   ├── requirements.txt             # Dependencies
│   └── README.md                    # Module-specific README
│
├── external/                        # External models and baselines
│   └── google-asl-250/              # ASL recognition baseline
│       ├── model.tflite             # Model weights
│       ├── evaluate_asl_250.py      # Evaluation script
│       ├── realtime_asl_250.py      # Real-time script
│       └── README.md                # Baseline documentation
│
├── Report_OP_LAB/                   # Academic report (LaTeX source)
│   ├── chapters/                    # Report sections
│   ├── figures/                     # Diagrams and plots
│   ├── backmatter/                  # References and appendix
│   ├── main.tex                     # Main LaTeX document
│   └── main.pdf                     # Compiled PDF (for reference)
│
├── Presentation_Demo/               # Presentation slides (Beamer)
│   ├── main.tex                     # Main presentation file
│   └── main.pdf                     # Compiled PDF (for reference)
│
├── docs/                            # Documentation (organized)
│   ├── architecture/                # System architecture docs
│   ├── setup/                       # Setup and installation guides
│   ├── evaluation/                  # Evaluation methodology
│   ├── INTEGRATION.md               # Integration pipeline doc
│   └── EXPERT_REVIEW.md             # Expert review notes
│
├── scripts/                         # Utility scripts
│   └── [automation and utility scripts]
│
├── README.md                        # Main project README
├── README_PROJECT.md                # Project overview
├── integrated_pipeline.py           # Main pipeline script
├── realtime_integrated.py           # Real-time processing script
├── requirements.txt                 # Global dependencies (if any)
├── .gitignore                       # Git ignore rules
└── LICENSE                          # License file
```

## File Naming Conventions

- **Python files**: `snake_case.py` (e.g., `data_processor.py`)
- **Directories**: `snake_case/` or `PascalCase/` for major modules
- **LaTeX files**: `section_name.tex` (e.g., `02_architecture.tex`)
- **Documentation**: `CAPS_OR_CamelCase.md`

## Code Organization Best Practices

### Python Modules
- Keep modules focused on single responsibility
- Use type hints for better IDE support
- Include docstrings for functions and classes
- Add unit tests in `tests/` directory

### Data Management
- Store datasets in `data/` or `dataset/` directories
- Document data format and preprocessing steps
- Include data versioning notes

### Model Files
- Store model weights with version information
- Maintain model cards with:
  - Architecture description
  - Training details
  - Performance metrics
  - Citation/attribution

## Git Workflow

### Before Committing
```bash
# Check for uncommitted changes
git status

# Review changes
git diff

# Stage changes
git add [files]

# Run tests (if applicable)
python -m pytest tests/

# Commit with clear message
git commit -m "Description of changes"
```

### Commit Message Format
```
[TYPE] Brief description (50 chars max)

Detailed explanation of changes (if needed)
- Bullet point 1
- Bullet point 2

Related to: [issue/feature] #123
```

**Types**: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

## Documentation Standards

- Use Markdown for all documentation
- Include code examples where relevant
- Add usage instructions for new modules
- Update README when adding features
- Document API changes

## Contribution Checklist

- [ ] Code follows project structure
- [ ] Python code formatted with consistent style
- [ ] Docstrings added to new functions
- [ ] Tests added/updated if applicable
- [ ] README and relevant docs updated
- [ ] .gitignore updated if new file types added
- [ ] No unnecessary build artifacts committed
- [ ] Commit messages are clear and descriptive

