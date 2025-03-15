# Cutting Edge Project Guidelines

## Build/Test/Lint Commands
```bash
# Activate environment
uv venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
pytest tests/test_specific_file.py  # Run single test file
pytest tests/test_specific_file.py::test_specific_function  # Run single test

# Code formatting
black .
isort .

# Linting
ruff check .
mypy .
```

## Code Style Guidelines
- **Imports**: Group imports: std lib, third-party, local (use isort)
- **Formatting**: Follow black (88 char line length)
- **Types**: Use type annotations for all functions and class attributes
- **Documentation**: Add docstrings for all functions and classes
- **Comments**: Add comments for complex logic, with references to papers/sources
- **Error Handling**: Use try/except blocks with specific exception types
- **ML Code**: Write simple, beginner-friendly code with detailed comments
- **Variable Names**: Use descriptive names (avoid abbreviations)