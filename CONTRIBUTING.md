# Contributing to ManufacturingIQ

Thank you for your interest in contributing to ManufacturingIQ! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. **Search existing issues** first to avoid duplicates
2. Use the bug report template when creating a new issue
3. Include:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Logs or error messages

### Suggesting Enhancements

1. **Search existing issues** first
2. Clearly describe the enhancement and its benefits
3. Include examples of how the feature would work

### Pull Requests

1. **Fork the repository** and create a feature branch from `main`
2. **Follow existing code style** (PEP 8 for Python, type hints, docstrings)
3. **Write tests** for new functionality
4. **Run the test suite** before submitting:
   ```bash
   pytest tests/ -v
   ```
5. **Keep PRs focused** on a single concern
6. **Update documentation** (README, docstrings, etc.) as needed
7. **Ensure the CI pipeline passes**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/ManufacturingIQ.git
cd ManufacturingIQ

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy

# Copy environment config
cp .env.example .env
# Edit .env with your settings
```

## Code Style

- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Type Hints**: Use type annotations for all function signatures
- **Docstrings**: Use Google-style docstrings
- **Imports**: Group in order: standard library, third-party, local modules
- **Formatting**: Use [ruff](https://github.com/astral-sh/ruff) for linting:
  ```bash
  ruff check .
  ```

## Testing

- All new features should include tests
- Tests go in the `tests/` directory
- Use descriptive test function names
- Run tests with:
  ```bash
  pytest tests/ -v --cov
  ```

## Commit Messages

Use conventional commit format:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `refactor:` code restructuring
- `test:` test additions/updates
- `chore:` maintenance tasks

Example: `feat: add per-user rate limiting with JWT auth`

## Questions?

Open a [discussion](https://github.com/deepmhatre13/ManufacturingIQ/discussions) or reach out to the maintainers.