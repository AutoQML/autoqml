# AutoQML Library

The project dependencies of `autoqml-lib` can be installed via `pip` or via `poetry`.

Poetry is the recommended installation method for research and development environments, as it ensures that all `autoqml-lib` developers and users share the exact same dependencies. It also simplifies the creation and activation of Python virtual environments.

Pip might be easier to use in containerized and/or automated installation environments, where a simple installation method without the requirement of a Poetry installation could be preferable.  


## Setup Guide with `pip`

1. Create and activate a Python virtual environment with Python 3.9 or Python 3.10. If you omit this step, the `autoqml-lib` dependencies will be installed in your system Python. This might make sense when installing in a Docker container or similar environments. However, for desktop installations, it is highly recommended to install the dependencies in a dedicated virtual environment.

2. `cd` into the `autoqml-lib` directory and install the dependencies into the active Python virtual environment using `pip`:

``` bash
$ pip install -r requirements.txt
```


## Setup Guide with Poetry

1. Install Poetry.
AutoQML uses Poetry to manage Python dependencies and Python virtual environments.
Poetry is multi-platform and should work equally well on Linux, macOS and Windows. If Poetry is not already installed on your machine, it needs to be installed first.
There are different ways to install Poetry. The official installer script with its installation instructions can be found on the Poetry website here: https://python-poetry.org/docs/#installing-with-the-official-installer
Usually, the "Linux, macOS and Windows (WSL)" installer is the right one to use.

2. Install AutoQML dependencies.
After Poetry has been installed, please `cd` into your `autoqml-lib` directory. Please install the project dependencies with the following command:

```bash
$ poetry install
```
A prerequisite for the dependency installation is the availability of Python 3.9 or 3.10 on your system's path, so that Poetry can use it to create the new virtual environment. The required Python versions can be found in `pyproject.toml` under `[tool.poetry.dependencies]`.

3. Activate the Python virtual environment for `autoqml-lib`.
Still in the project directory, please activate the Python virtual environment using Poetry:

``` bash
$ poetry shell
```

Further information about the basic usage of Poetry can be found here: https://python-poetry.org/docs/basic-usage/

4. Start JupyterLab IDE.
To start working with the `autoqml-lib`

5. Finalize your `autoqml-lib` work session.
To deactivate the Python virtual environment (after finishing your work session with JupyterLab and `autoqml-lib`), please enter this shell command:
  
``` bash
$ exit
```

This exits the nested shell, which was previously created via `$ poetry shell`.


## Generate `requirements.txt` for `pip` from Poetry 

Export the project dependencies to a `requirements.txt` file using Poetry:

``` bash
$ poetry export -f requirements.txt --output requirements.txt
```

If necessary, install the Poetry plugin `poetry-plugin-export`:

``` bash
$ poetry self add poetry-plugin-export
```
