# Contributing Guide

## Team Members & Roles

| Member | Responsibilities |
|--------|-----------------|
| Member 1 | Data pipeline, model training |
| Member 2 | API |
| Member 3 | Monitering |

> **Note**: Update the table above with actual team member names and GitHub usernames.

## Development Workflow

### Branch Strategy
- `main` — Production-ready code (protected)
- `develop` — Integration branch
- `feature/<name>` — Feature branches
- `fix/<name>` — Bug fix branches

### Workflow
1. Create a feature branch from `develop`
2. Implement changes with meaningful commits
3. Write/update tests for new functionality
4. Open a Pull Request to `develop`
5. At least one team member reviews the PR
6. After CI passes, merge the PR
7. Periodically merge `develop` into `main`

### Commit Messages
Follow conventional commits:
```
feat: add EfficientNet model architecture
fix: correct class weight calculation for imbalanced data
test: add integration tests for prediction endpoint
docs: update README with deployment instructions
ci: add Docker build step to pipeline
```

## Setup for Development

```bash
# Clone
git clone https://github.com/tmt2504/medical-image-classification.git
cd medical-image-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run linting
flake8 src/ tests/ --max-line-length=100
black --check src/ tests/
isort --check-only src/ tests/
```

## Code Style

- **Formatter**: Black (line length: 100)
- **Import sorter**: isort
- **Linter**: Flake8
- **Type hints**: Encouraged for public interfaces
- **Docstrings**: Google style for all public functions

## Individual Contribution Tracking

Each team member should:
1. Use their own GitHub account for commits
2. Reference issue numbers in commit messages
3. Document their specific contributions in PR descriptions
4. Keep this file updated with their role details
