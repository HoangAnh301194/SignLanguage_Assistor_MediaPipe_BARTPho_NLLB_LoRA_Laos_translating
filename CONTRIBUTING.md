# Contributing to the Vietnamese Sign Language Project

## Getting Started

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install module dependencies** (if working on specific modules)
   ```bash
   cd VSL_Vietnamese_NLP
   pip install -r requirements.txt
   
   cd ../dich-vi_lo
   pip install -r requirements.txt
   ```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/bug-description
```

### 2. Make Your Changes
- Follow the project structure guidelines (see [docs/PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md))
- Write clean, documented code
- Add unit tests for new functionality
- Update relevant documentation

### 3. Test Your Changes
```bash
# Run module-specific tests
cd VSL_Vietnamese_NLP
python -m pytest tests/

# Or test integration
python ../integrated_pipeline.py --test
```

### 4. Commit Your Work
```bash
git add .
git commit -m "[TYPE] Brief description of changes"
# Examples:
# git commit -m "feat: Add new gloss-to-sentence model"
# git commit -m "fix: Correct alignment issue in pipeline"
# git commit -m "docs: Update installation instructions"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
# Then create PR on GitHub/GitLab
```

## Code Style Guidelines

### Python
- **PEP 8** compliance required
- Use type hints for function signatures
- Max line length: 100 characters
- Use meaningful variable names

Example:
```python
from typing import List, Dict, Optional

def process_glosses(glosses: List[str], 
                   confidence: float = 0.5) -> Dict[str, any]:
    """
    Process recognized glosses into structured format.
    
    Args:
        glosses: List of recognized sign glosses
        confidence: Confidence threshold for filtering
        
    Returns:
        Processed gloss dictionary with metadata
    """
    # Implementation
    pass
```

### LaTeX
- Use consistent formatting
- Comment structure clearly
- Label figures and tables appropriately
- Reference figures/tables explicitly in text

### Documentation (Markdown)
- Use clear headings hierarchy
- Include code examples with syntax highlighting
- Add table of contents for long documents
- Use bullet points for lists

## Key Modules & Contact Points

### VSL Recognition
- **Location**: `external/asl_baseline/`
- **Key Files**: `realtime_asl_250.py`
- **Maintainer**: [Name/Email]

### Vietnamese NLP
- **Location**: `VSL_Vietnamese_NLP/`
- **Key Files**: `src/nlp/sentence_builder.py`
- **Maintainer**: [Name/Email]

### Translation Module
- **Location**: `dich-vi_lo/`
- **Key Files**: `translate.py`
- **Maintainer**: [Name/Email]

## Testing Requirements

- **Unit tests**: Required for all new features
- **Integration tests**: Required for pipeline changes
- **Test coverage**: Aim for >80% coverage
- **Test location**: `tests/` directory within each module

### Running Tests
```bash
# Run all tests
pytest

# Run specific module tests
pytest VSL_Vietnamese_NLP/tests/

# Run with coverage
pytest --cov=src tests/
```

## Documentation Requirements

When submitting changes:
- [ ] Update relevant README files
- [ ] Add/update docstrings
- [ ] Update [docs/PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) if structure changes
- [ ] Include usage examples for new features
- [ ] Update [docs/INTEGRATION.md](./INTEGRATION.md) if affecting pipeline

## Commit Message Format

```
[TYPE] Brief description (50 chars max)

Detailed explanation (if needed)
- Change 1
- Change 2

Fixes: #issue_number
Related to: #other_issue
```

**Types**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code restructuring (no functional change)
- `test:` Test additions/modifications
- `chore:` Maintenance tasks
- `perf:` Performance improvements

## Pull Request Process

1. **Before submitting**:
   - Run tests: `pytest`
   - Check style: `flake8` or `pylint`
   - Update documentation
   - Verify .gitignore is respected

2. **PR Description**:
   ```markdown
   ## Description
   Brief description of what this PR does

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring

   ## Related Issues
   Fixes #123

   ## Testing Done
   - [ ] Unit tests passed
   - [ ] Integration tests passed
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No new warnings generated
   ```

## Issues and Feature Requests

- **Report bugs**: Include reproduction steps and environment info
- **Feature requests**: Explain use case and expected behavior
- **Discussions**: Use for design decisions and major changes

## Research Paper & Citation

If using this code in research:
```bibtex
@software{vsl_project_2026,
  title={Vietnamese Sign Language Recognition and Translation System},
  author={[Authors]},
  year={2026},
  url={[Repository URL]}
}
```

## Questions?

- Check existing documentation in `docs/`
- Review module READMEs for specifics
- Open an issue for clarification
- Contact maintainers directly if urgent

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🎉

