# Contributing to AutoQML

We welcome contributions to the AutoQML library! This guide will help you get started with development.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/yourusername/autoqml.git
   cd autoqml
   ```

2. Install development dependencies with Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Code Style

We follow these coding standards:
- PEP 8 for Python code style
- Type hints for function parameters and returns
- Docstrings for all public functions and classes

## Testing

1. Run tests using pytest:
   ```bash
   pytest tests/
   ```

2. Ensure test coverage:
   ```bash
   pytest --cov=autoqml tests/
   ```

3. Write tests for new features:
   - Unit tests for individual components
   - Integration tests for pipelines
   - Test both success and failure cases

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request:
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all tests pass
   - Update documentation as needed

## Documentation

Update relevant documentation:
   - API documentation for new features
   - Usage examples
   - README.md if necessary

## Code Review Process

1. All code changes require review
2. Address reviewer comments promptly
3. Keep pull requests focused and manageable
4. Ensure CI/CD checks pass

## Questions?

If you have questions about contributing:
1. Check existing issues
2. Create a new issue for discussion
3. Reach out to maintainers
