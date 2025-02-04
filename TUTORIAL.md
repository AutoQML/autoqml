# AutoQML Library Tutorial
## Part 1: Training and saving (quantum) machine learning pipelines with the AutoQML library

This beginner-friendly tutorial will guide you through using the `autoqml` Python library, a powerful tool designed for automating the tuning of classical as well as quantum machine learning model pipelines. 
The tutorial is based on the KEB use case. The demo notebooks for all AutoQML use cases, including the KEB use case, follow the same pattern explained here. The demo notebooks can be found in their respective subdirectories under the `use_cases` directory.
This is the first of two parts of the tutorial. By the end of this first part, you will be able to load data, visualize it, train models, and save your quantum machine learning pipelines using a Jupyter Notebook environment.

## Setting Up Your Environment
### Step 0: Prerequisites:
- Basic familiarity with Python programming
- Installation of Jupyter Lab (or alternatively: Jupyter Notebook)
- Installation of the `autoqml` library and its dependencies. 

For more details about the above mentioned installations please check out the `README.md` file.
     
### Step 1: Setting Up Your Poetry Environment

First, you need to ensure that you have the Poetry environment for the `autoqml` project installed as described in the `README.md` file. As a result, JupyterLab should be installed and running. 

Start your JupyterLab:

```bash
$ jupyter lab
```

This command will open JupyterLab in your default web browser.

### Step 2: Importing Necessary Libraries

Create a new Python notebook and start by importing the required libraries. The AutoQML library name is `autoqml`. In the first cell of your notebook, type and run:

```python
import autoqml
import pandas as pd
import matplotlib.pyplot as plt
```

This code block imports the `autoqml` library for quantum machine learning, `pandas` for data manipulation, and `matplotlib` for data visualization. In later steps of the tutorial you will load some necessary classes from the `autoqml` library.

## Using AutoQML Pipelines
### Step 3: Loading and Visualizing Data

To effectively train a quantum machine learning model, you will need a dataset. For the purposes of this tutorial, let's assume you have a CSV file named `data.csv` containing your dataset, consisting of timeseries features `X` and labels `y`. For a fully detailed source code example, please consult the `training.ipynb` notebooks in the `use_cases` directory, for example the `use_cases/keb/training.ipynb` notebook.

#### Loading Data

In a new cell, load your data:

```python
from sklearn.model_selection import train_test_split

# Load your data from a source file using your own
# data loading function
X, y = your_data_loading_function('data.csv')

# Split the dataset in training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=777,
    stratify=y,
)
```

#### Visualizing Data

Probably you will want to visualize some aspects of your data to understand its distribution, possible correlations, and other insights. One possible way to do this is shown in the following Python code cell:

```python
# Display the first few rows of the datasets
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())

# Plotting the feature from the train dataset. 
# Use similar code to plot further data of interest:
plt.figure(figsize=(10, 5))
plt.hist(X_train, bins=50, alpha=0.75)
plt.title('Feature Distribution')
plt.xlabel('Feature')
plt.ylabel('Frequency')
plt.show()
```


### Step 4: Training with autoqml

Now, it's time to train a quantum machine learning model using `autoqml`.

#### Initialize and Configure the AutoQML Library

In your notebook, set up the AutoQML environment. The following example shows how a custom configuration might be used to limit the classification choices to only Quantum Support Vector Machine (QSVC), the RescalingChoice to MinMaxScaler, to turn off the Dimensionality Reduction, and to set some hyperparameters for the QSVC classifier.

This example configuration includes a timeout of 300 seconds. This means that the optimizer will use a maximum of 300 seconds and will stop the optimization of the pipeline after that. The timeout of 300 seconds does not include the startup time for loading the data and algorithms.

If no such custom configuration is provided (if the `custom_config` `dict` is empty), the AutoQML uses the entire available search space for finding the optimal pipeline configuration for your machine learning problem, including quantum as well as classical algorithms.

Please note that it is your decision as user to instantiate the correct AutoQML class, corresponding to the machine learning problem you are solving: `AutoQMLTabularClassification`, `AutoQMLTimeSeriesClassification`, `AutoQMLTabularRegression`.

In the following example, the user wants to solve a timeseries classification task on his data and decides to instantiate the AutoQML-Lib class `AutoQMLTimeSeriesClassification`.

```python
from autoqml.automl import AutoQMLTimeSeriesClassification
from autoqml.messages import AutoQMLFitCommand

custom_config = {
    'autoqml.search_space.classification.ClassificationChoice__choice':
        'qsvc',
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice':
        'min_max_scaling',
    'autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice':
        'no-op',
    'autoqml.search_space.classification.quantum.qsvc.QSVC__num_repetitions':
        4,
    'autoqml.search_space.classification.quantum.qsvc.QSVC__num_qubits':
        3,
    'autoqml.search_space.classification.quantum.qsvc.QSVC__C':
        100,
}

# Initialize the AutoQML Pipeline
autoqml_pipeline = AutoQMLTimeSeriesClassification()

# The fit command contains the training data as well as 
# configuration parameters for the pipeline.
cmd = AutoQMLFitCommand(
    X_train,
    y_train,
    timedelta(seconds=300),
    configuration=custom_config,
)
```

#### Optimize the Pipeline and Train the Model(s)

Using `autoqml` to automatically find a good machine learning pipeline for your (quantum computing) machine learning projects is easy: In only one line of code you can create and fit an optimized pipeline for your machine learning task. In the following example, input features and target detail are contained in the `cmd` object, which was created in the previous section:

```python
autoqml_pipeline.fit(cmd)
```

### Step 5: Saving the Pipeline

Once the training is finished, save the pipeline using the `dill` library, which is automatically installed with AutoQML-Lib:

```python
import dill

with open('path_to_your_autoqml_pipeline.dill', 'wb') as file:
    dill.dump(autoqml_pipeline, file=file)
```

This will save your trained quantum machine learning pipeline to a file called `my_quantum_ml_pipeline.dill`, which you can load later to make predictions on new data. `dill` is a library which can save (and later load) all trained components of the pipeline to disk.

# Part 2: Evaluating and comparing machine learning pipelines with the AutoQML library

The second part of this tutorial will help you to understand how to evaluate and compare different AutoQML pipelines using the `autoqml` library in a Jupyter Notebook environment. You’ll learn how to load pre-trained models and compare their performance.
Please note, that it is recommended to create a new Jupyter Notebook for the second part.

### Step 1: Import Libraries

In the first cell, you should import necessary Python libraries. For example use the following code:

```python
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
```

### Step 3: Load and Visualize Data

The next step involves loading your dataset and visualizing it to better understand its structure and distribution. Split the data into training and testing datasets and use the test data set for model evaluation. If the data is the same as in the first part of the tutorial, you can re-use the same code from Part 1, Step 3 and Step 4 here.

### Step 4: Load and Evaluate autoqml Pipelines

You can load and evaluate multiple pre-trained autoqml pipelines as follows. Here we load for example the pre-trained pipeline from Part 1.

#### Load a Pre-trained Pipeline

```python
# Load a pre-trained AutoQML pipeline
with open('path_to_your_autoqml_pipeline.dill', 'rb') as file:
    automl_pipeline = dill.load(file)
```

#### Predict and Evaluate

```python
# Make predictions
y_pred = automl_pipeline.predict(X_test)

# Plot the predictions
plt.hist(y_pred, bins=20, alpha=0.75)
plt.title('Prediction Distribution')
plt.xlabel('Predicted Values')
plt.ylabel('Frequency')
plt.show()

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))

# Display a confusion matrix (relevant for classification pipelines)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

### Step 5: Comparison of Different Pipelines

Repeat Step 4 for different pre-trained AutoQML pipelines (`automl_pipeline_2`, `automl_pipeline_3`, etc.) and compare their performance.

```python
# Compare accuracy and balanced accuracy scores from different pipelines
print('Accuracy Pipeline 1:', accuracy_score(y_test, y_pred_1))
print('Accuracy Pipeline 2:', accuracy_score(y_test, y_pred_2))
```

### Conclusion

Congratulations! You've just completed a beginner-level tutorial on using the `autoqml` library through a Jupyter Notebook. You can now load and visualize data, train quantum machine learning models, and save your pipelines for future use. You have learned how to load, visualize, and evaluate multiple AutoQML pipelines using a Jupyter Notebook. Experiment with different parameters and datasets to enhance your understanding of AutoQML’s capabilities.

### What's Next?

- Explore more features of the `autoqml` library.
- Try training with different datasets.
- Learn more about quantum computing and its applications in machine learning.
- Fine-tune these models based on your specific datasets.
- Explore further by adjusting the pipeline components and their parameters.
- Implement more complex quantum machine learning strategies as your familiarity with the `autoqml` library grows.

Happy Automated Quantum Machine Learning!
