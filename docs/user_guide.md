# AutoQML User Guide

## Pipeline Configuration

### Custom Configurations

You can customize your pipeline using configuration dictionaries. For detailed parameter descriptions, see the [API Reference](./api_reference.md).

Example configuration:

```python
custom_config = {
    'autoqml_lib.search_space.classification.ClassificationChoice__choice': 'qsvc',
    'autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice': 'min_max_scaling'
}
```

### Component Overview

#### Preprocessing

- **Scaling**:
  - MinMaxScaler
  - StandardScaler
  - RobustScaler
  - QuantileTransformer
- **Dimensionality Reduction**:
  - PCA
  - Kernel PCA
  - t-SNE
  - UMAP

See [API Reference - Preprocessing](./api_reference.md#preprocessing-components) for details.

#### Models

- **Quantum Classification**:
  - QSVC (Quantum Support Vector Classification)
  - QNN Classifier (Quantum Neural Network)
  - QKNeighbors Classifier
  - QRandomForest Classifier
- **Quantum Regression**:
  - QNNRegressor (Quantum Neural Network)
  - QRCRegressor (Quantum Reservoir Computing)
  - QKRR (Quantum Kernel Ridge Regression)
  - QSVR (Quantum Support Vector Regression)
  - QGPR (Quantum Gaussian Process Regression)
- **Classical Classification Models**:
  - Decision Tree Classifier
  - Logistic Regression Classifier
  - Ridge Classifier
  - Perceptron
  - SVC (Support Vector Classification)
  - Random Forest Classifier
- **Classical Regression Models**:
  - Linear Regression
  - Kernel Ridge Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - SVR (Support Vector Regression)
  - Gaussian Process Regression
  - NNR (Neural Network Regressor)

See [API Reference - Models](./api_reference.md#classification-components) for details.

#### Utilities

- **Data Handling**:
  - Quantum Data Loaders
  - Batch Generators
  - Cross Validators
- **Metrics**:
  - Classification Metrics
  - Regression Metrics
  - Quantum State Fidelity

## Training Models

### Basic Training

For basic training examples, see [Examples - Basic Usage](./examples.md#basic-usage-examples).

### Early Stopping

   ```python
   cmd = AutoQMLFitCommand(
       X_train, y_train,
       timedelta(seconds=300),
       early_stopping_rounds=10
   )
   ```

## Model Persistence

### Saving Models

```python
import dill

with open('model.dill', 'wb') as f:
    dill.dump(pipeline, f)
```

### Loading Models

```python
with open('model.dill', 'rb') as f:
    loaded_pipeline = dill.load(f)
```

## Best Practices

1. **Data Preparation**:

   - AutoQML handles data cleaning and normalization.
   - AutoQML can impute missing values
   - However, it's recommended to clean and preprocess your data before training

2. **Model Selection**:

   - Start with simpler pipelines, for example by using a custom configuration with a single, simple choice for each component
   - Gradually increase complexity by setting more complex component choices and/or removing steps from the custom configuration
   - Monitor computational resources

3. **Hyperparameter Optimization**:

   - Define reasonable parameter ranges
   - Use appropriate optimization metrics
   - Set realistic time budgets
   - The more options AutoQML has to choose from, the longer the optimization will take to produce good results

4. **Production Deployment**:
   - Implement proper error handling
   - Monitor model performance
   - Plan for model updates

## Troubleshooting

For common issues and solutions, see the [Troubleshooting Guide](./troubleshooting.md).
