# Getting Started with AutoQML

## Installation

AutoQML can be installed using either `pip` or `poetry` (recommended).

### Using Poetry (Recommended)

1. Install Poetry if not already installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install AutoQML dependencies:
   ```bash
   cd autoqml-lib
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Using Pip

1. Create and activate a Python virtual environment (Python 3.9 or 3.10)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Concepts

### Pipeline Components

AutoQML pipelines consist of three main components:
1. **Preprocessing**: Data scaling, dimensionality reduction, and feature selection
2. **Model Selection**: Both quantum and classical models
3. **Hyperparameter Optimization**: Automated tuning using Optuna

### Configuration System

Pipelines are configured using a dictionary-based configuration system. See the [User Guide](./user_guide.md#pipeline-configuration) for details.

## Next Steps

1. Check out the [Examples](./examples.md) for practical usage scenarios
2. Read the [User Guide](./user_guide.md) for detailed usage instructions
3. Review the [API Reference](./api_reference.md) for component details
4. See [Troubleshooting](./troubleshooting.md) if you encounter issues

## Version Compatibility

For version information and recent changes, see our [Changelog](./changelog.md).
