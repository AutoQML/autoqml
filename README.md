# AutoQML Library

AutoQML is a framework that adapts the principles of Automated Machine Learning (AutoML) to the domain of Quantum Machine Learning (QML). By automating the processes of hyperparameter optimization and pipeline construction, AutoQML reduces the need for manual intervention, making QML more accessible. It leverages the QML library sQUlearn, offering a modular and unified interface to support various QML algorithms and construct end-to-end pipelines for supervised learning tasks. AutoQML is evaluated on industrial use cases, demonstrating its capability to generate effective QML pipelines that are competitive with both classical ML models and manually crafted quantum solutions.

## Installation

`autoqml` can be installed via `pip` or via `poetry`.

### Quick install

For simple application of `autoqml`, it is sufficient to create and activate a Python virtual environment with Python 3.9 or Python 3.10 and then install it with

``` bash
$ pip install .
```

for local clones, or directly from github with

``` bash
$ pip install git+https://github.com/AutoQML/autoqml.git
```

### Development Install

Poetry is the recommended installation method for research and development environments, as it ensures that all `autoqml` developers and users share the exact same dependencies. It also simplifies the creation and activation of Python virtual environments.

Pip might be easier to use in containerized and/or automated installation environments, where a simple installation method without the requirement of a Poetry installation could be preferable.  


#### Setup Guide with `pip`

1. Create and activate a Python virtual environment with Python 3.9 or Python 3.10. If you omit this step, the `autoqml` dependencies will be installed in your system Python. This might make sense when installing in a Docker container or similar environments. However, for desktop installations, it is highly recommended to install the dependencies in a dedicated virtual environment.

2. `cd` into the `autoqml` directory and install the dependencies into the active Python virtual environment using `pip`:

``` bash
$ pip install -r requirements.txt
```


#### Setup Guide with Poetry

1. Install Poetry.
AutoQML uses Poetry to manage Python dependencies and Python virtual environments.
Poetry is multi-platform and should work equally well on Linux, macOS and Windows. If Poetry is not already installed on your machine, it needs to be installed first.
There are different ways to install Poetry. The official installer script with its installation instructions can be found on the Poetry website here: https://python-poetry.org/docs/#installing-with-the-official-installer
Usually, the "Linux, macOS and Windows (WSL)" installer is the right one to use.

2. Install AutoQML dependencies.
After Poetry has been installed, please `cd` into your `autoqml` directory. Please install the project dependencies with the following command:

```bash
$ poetry install
```
A prerequisite for the dependency installation is the availability of Python 3.9 or 3.10 on your system's path, so that Poetry can use it to create the new virtual environment. The required Python versions can be found in `pyproject.toml` under `[tool.poetry.dependencies]`.

3. Activate the Python virtual environment for `autoqml`.
Still in the project directory, please activate the Python virtual environment using Poetry:

``` bash
$ poetry shell
```

Further information about the basic usage of Poetry can be found here: https://python-poetry.org/docs/basic-usage/

4. Start JupyterLab IDE.
To start working with the `autoqml`

5. Finalize your `autoqml` work session.
To deactivate the Python virtual environment (after finishing your work session with JupyterLab and `autoqml`), please enter this shell command:
  
``` bash
$ exit
```

This exits the nested shell, which was previously created via `$ poetry shell`.


#### Generate `requirements.txt` for `pip` from Poetry 

Export the project dependencies to a `requirements.txt` file using Poetry:

``` bash
$ poetry export -f requirements.txt --output requirements.txt
```

If necessary, install the Poetry plugin `poetry-plugin-export`:

``` bash
$ poetry self add poetry-plugin-export
```

## Getting Started with AutoQML

### Basic Concepts

#### Pipeline Components

AutoQML pipelines consist of three main components:
1. **Preprocessing**: Data scaling, dimensionality reduction, and feature selection
2. **Model Selection**: Both quantum and classical models
3. **Hyperparameter Optimization**: Automated tuning using Optuna

#### Configuration System

Pipelines are configured using a dictionary-based configuration system. See the [User Guide](./user_guide.md#pipeline-configuration) for details.

### Next Steps

1. Check out the [Examples](./examples.md) for practical usage scenarios
2. Read the [User Guide](./user_guide.md) for detailed usage instructions
3. Review the [API Reference](./api_reference.md) for component details
4. See [Troubleshooting](./troubleshooting.md) if you encounter issues

### Version Compatibility

For version information and recent changes, see our [Changelog](./changelog.md).
