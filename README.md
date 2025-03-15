# Cutting Edge Problem

AI-powered cutting stock optimization for sewing patterns.

## Installation

```bash
uv venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
uv pip install -e ".[dev]"
```

To exit the environment:

```bash
deactivate
```

## Usage

```python
python src/cutting_edge/main.py
```

## Development

1. Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

2. Run tests:

```bash
pytest
```

3. Format code:

```bash
black .
isort .
```

4. Run linter:

```bash
ruff check .
```
