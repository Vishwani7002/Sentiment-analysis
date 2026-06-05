# Contributing

Thank you for your interest in contributing to this project.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Code Style

- Follow PEP 8 conventions.
- Add docstrings to all public functions and classes.
- Keep functions small and single-purpose.

## Testing

Run the test suite before submitting a pull request:

```bash
pytest tests/ -v
```

All existing tests must pass. Add new tests for any new functionality.

## Submitting a Pull Request

1. Commit your changes with a clear, descriptive message:
   ```
   feat: add ONNX export for faster inference
   fix: handle empty string input in dataset.py
   ```
2. Push to your fork and open a Pull Request against `main`.
3. Describe what your PR does and why in the PR description.

## Reporting Issues

Open a GitHub Issue with:
- What you expected to happen.
- What actually happened.
- Steps to reproduce.
- Your Python version and OS.
