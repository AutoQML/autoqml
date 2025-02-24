# AutoQML API Reference

## Core Concepts

### Configuration System

The AutoQML configuration system allows you to customize every aspect of your machine learning pipeline through a unified dictionary-based interface.

#### Configuration Key Format

Configuration keys follow this format: `module.path.ClassName__parameter`

For example:

```python
'autoqml.search_space.regression.RegressionChoice__choice': 'svr'
```

#### Configuration Dictionary Structure

The configuration dictionary is a flat key-value store where:

- Keys are fully qualified parameter paths
- Values are the parameter settings

Example:

```python
custom_config = {
    'autoqml.search_space.regression.RegressionChoice__choice': 'svr',
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice': 'standard_scaling',
}
```

#### Default Values and Overrides

- Each component has default values that are used for initialization if not specified. If no choice is specified for a component of the pipeline, AutoQML Lib will automatically generate multiple pipelines with different component choices. Then the best performing pipeline is selected based on the evaluation metrics.
- If no component choice is specified at all, AutoQML Lib will have a much greater search space to create automatically even more pipelines and select the best performing one based on the evaluation metrics.
- Optimization is done based on the evaluation metrics and the time needed will vary depending on the complexity of the problem and how many pipelines are created and evaluated against each other based on the evaluation metrics.
- You can override any parameter by including it in your configuration
- For optimal results, only override the parameters that you want to change, and provide sufficient time for the optimization.

### Pipeline Components

AutoQML uses a modular pipeline architecture composed of:

#### Estimator Choices

Base components that implement the core machine learning algorithms:

- Classical algorithms (SVM, Random Forest, etc.)
- Quantum algorithms (QSVM, QNN, etc.)

#### Tunable Components

Components that can be automatically tuned:

- Preprocessing steps
- Model hyperparameters
- Pipeline structure

#### Pipeline Construction

Pipelines are constructed automatically based on:

- The task type (classification, regression)
- Data characteristics
- Your configuration settings

## Search Space Configuration

### Task-Specific Pipelines

#### Tabular Regression Pipeline

For regression tasks on tabular data.

**Base Class**: `AutoQMLTabularRegression`

**Example Configuration**:

```python
config = {
    'autoqml.search_space.regression.RegressionChoice__choice': 'svr',
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice': 'standard_scaling',
}
```

#### Tabular Classification Pipeline

For classification tasks on tabular data.

**Base Class**: `AutoQMLTabularClassification`

**Example Configuration**:

```python
config = {
    'autoqml.search_space.classification.ClassificationChoice__choice': 'random_forest_classifier',
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice': 'standard_scaling',
}
```
#### Time Series Classification Pipeline

For classification tasks on time series data.

**Base Class**: `AutoQMLTimeSeriesClassification`

**Example Configuration**:

```python
config = {
    'autoqml.search_space.classification.ClassificationChoice__choice': 'random_forest_classifier',
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice': 'standard_scaling',
}
```

### Component Categories

#### 1. Data Loading Components

##### TabularizeTimeSeries

**Class**: `TabularizeTimeSeries`
**Module**: `autoqml.search_space.data_loading.timeseries.tabularize`

A specialized component for converting time series data into a tabular format. This component is designed for specific use cases and may require source code modification for custom requirements.

**Note**: This is a lower-level component that does not follow the standard AutoQML configuration system.

| Parameter   | Type | Default | Description                      |
| ----------- | ---- | ------- | -------------------------------- |
| tile_size_x | int  | None    | Size of tiles in the x dimension |
| tile_size_y | int  | None    | Size of tiles in the y dimension |

**Usage Notes**:

- Transforms time series data into a tabular format by tiling
- Handles both input data (X) and target values (y)
- Current implementation cuts data to fit tile sizes perfectly
- Returns transformed data as pandas DataFrame/Series

**Limitations**:

- No automatic parameter configuration
- May truncate data to fit tile sizes
- Specialized for specific time series formats
- May need code modifications for different use cases

#### 2. Data Cleaning Components

##### Imputation Strategies

###### ConstantImputation

**Class**: `ConstantImputation`
**Module**: `autoqml.search_space.data_cleaning.imputation.constant`

| Parameter  | Config Key                  | Type | Default | Range/Choices | Description                 |
| ---------- | --------------------------- | ---- | ------- | ------------- | --------------------------- |
| fill_value | `ConstantImputation__value` | Any  | 0       | -             | Value to use for imputation |

###### MeanImputation

**Class**: `MeanImputation`
**Module**: `autoqml.search_space.data_cleaning.imputation.mean`

| Parameter | Config Key                 | Type | Default | Range/Choices      | Description         |
| --------- | -------------------------- | ---- | ------- | ------------------ | ------------------- |
| strategy  | `MeanImputation__strategy` | str  | 'mean'  | ['mean', 'median'] | Imputation strategy |

###### DropImputation

**Class**: `DropImputation`
**Module**: `autoqml.search_space.data_cleaning.imputation.drop`

| Parameter | Config Key                  | Type  | Default | Range/Choices | Description             |
| --------- | --------------------------- | ----- | ------- | ------------- | ----------------------- |
| threshold | `DropImputation__threshold` | float | 0.5     | [0.0, 1.0]    | Missing value threshold |

#### 3. Preprocessing Components

##### Feature Encoding

###### CategoricalEncoder

**Class**: `CategoricalEncoder`
**Module**: `autoqml.search_space.preprocessing.encoding.categorical`

| Parameter      | Config Key                           | Type  | Default | Range/Choices | Description                              |
| -------------- | ------------------------------------ | ----- | ------- | ------------- | ---------------------------------------- |
| max_categories | `CategoricalEncoder__max_categories` | int   | 20      | [2, 100]      | Maximum number of categories to encode   |
| min_frequency  | `CategoricalEncoder__min_frequency`  | float | 0.01    | [0.0, 1.0]    | Minimum frequency for category inclusion |

###### OneHotEncoder

**Class**: `OneHotEncoder`
**Module**: `autoqml.search_space.preprocessing.encoding.one_hot`

| Parameter      | Config Key                      | Type  | Default | Range/Choices | Description                              |
| -------------- | ------------------------------- | ----- | ------- | ------------- | ---------------------------------------- |
| max_categories | `OneHotEncoder__max_categories` | int   | 20      | [2, 100]      | Maximum number of categories to encode   |
| min_frequency  | `OneHotEncoder__min_frequency`  | float | 0.01    | [0.0, 1.0]    | Minimum frequency for category inclusion |

##### Dimensionality Reduction

###### Autoencoder

**Class**: `Autoencoder`
**Module**: `autoqml.search_space.preprocessing.dim_reduction.autoencoder`

| Parameter     | Config Key                   | Type | Default    | Range/Choices                             | Description                           |
| ------------- | ---------------------------- | ---- | ---------- | ----------------------------------------- | ------------------------------------- |
| latent_dim    | `Autoencoder__latent_dim`    | int  | 10         | [2, input_dim]                            | Dimension of latent space             |
| act_fn        | `Autoencoder__act_fn`        | str  | 'GELU'     | ['ReLU', 'LeakyReLU', 'GELU', 'Tanh']     | Activation function for hidden layers |
| output_act_fn | `Autoencoder__output_act_fn` | str  | 'Tanh'     | ['Tanh']                                  | Output layer activation function      |
| loss_fn       | `Autoencoder__loss_fn`       | str  | 'mse_loss' | ['mse_loss', 'l1_loss', 'smooth_l1_loss'] | Loss function for training            |
| max_epochs    | `Autoencoder__max_epochs`    | int  | 50         | [10, 1000]                                | Maximum training epochs               |

###### PCA

**Class**: `PCA`
**Module**: `autoqml.search_space.preprocessing.dim_reduction.pca`

| Parameter    | Config Key          | Type | Default | Range/Choices   | Description                  |
| ------------ | ------------------- | ---- | ------- | --------------- | ---------------------------- |
| n_components | `PCA__n_components` | int  | None    | [1, n_features] | Number of components to keep |
| whiten       | `PCA__whiten`       | bool | False   | [True, False]   | Whether to whiten the data   |

###### TSNE

**Class**: `TSNE`
**Module**: `autoqml.search_space.preprocessing.dim_reduction.tsne`

| Parameter    | Config Key           | Type  | Default | Range/Choices | Description          |
| ------------ | -------------------- | ----- | ------- | ------------- | -------------------- |
| n_components | `TSNE__n_components` | int   | 2       | [1, 3]        | Number of dimensions |
| perplexity   | `TSNE__perplexity`   | float | 30.0    | [5.0, 50.0]   | Perplexity parameter |

###### UMAP

**Class**: `UMAP`
**Module**: `autoqml.search_space.preprocessing.dim_reduction.umap`

| Parameter   | Config Key          | Type  | Default | Range/Choices | Description                |
| ----------- | ------------------- | ----- | ------- | ------------- | -------------------------- |
| n_neighbors | `UMAP__n_neighbors` | int   | 15      | [2, 100]      | Number of neighbors        |
| min_dist    | `UMAP__min_dist`    | float | 0.1     | [0.0, 0.99]   | Minimum distance parameter |

##### Rescaling

###### MinMaxScaling

**Class**: `MinMaxScaling`
**Module**: `autoqml.search_space.preprocessing.rescaling.min_max_scaling`

| Parameter     | Config Key                     | Type  | Default | Range/Choices | Description        |
| ------------- | ------------------------------ | ----- | ------- | ------------- | ------------------ |
| feature_range | `MinMaxScaling__feature_range` | tuple | (0, 1)  | -             | Output value range |

###### MinMaxScalingForQuantumKernel

**Class**: `MinMaxScalingForQuantumKernel`
**Module**: `autoqml.search_space.preprocessing.rescaling.min_max_scaling`

| Parameter     | Config Key                                     | Type  | Default     | Range/Choices | Description        |
| ------------- | ---------------------------------------------- | ----- | ----------- | ------------- | ------------------ |
| feature_range | `MinMaxScalingForQuantumKernel__feature_range` | tuple | (-π/2, π/2) | -             | Output value range |

###### Normalization

**Class**: `Normalization`
**Module**: `autoqml.search_space.preprocessing.rescaling.normalization`

| Parameter | Config Key            | Type | Default | Range/Choices       | Description        |
| --------- | --------------------- | ---- | ------- | ------------------- | ------------------ |
| norm      | `Normalization__norm` | str  | 'l2'    | ['l1', 'l2', 'max'] | Normalization type |

###### StandardScaling

**Class**: `StandardScaling`
**Module**: `autoqml.search_space.preprocessing.rescaling.standard_scaling`

| Parameter | Config Key                   | Type | Default | Range/Choices | Description            |
| --------- | ---------------------------- | ---- | ------- | ------------- | ---------------------- |
| with_mean | `StandardScaling__with_mean` | bool | True    | [True, False] | Center the data        |
| with_std  | `StandardScaling__with_std`  | bool | True    | [True, False] | Scale to unit variance |

## Classification Components

Components for building classification models, including both classical and quantum classifiers.

#### Classical Classifiers

##### DecisionTreeClassifier

**Class**: `DecisionTreeClassifier`
**Module**: `autoqml.search_space.classification.classic.decision_tree_classifier`

| Parameter         | Config Key                                  | Type  | Default | Range/Choices                   | Description                          |
| ----------------- | ------------------------------------------- | ----- | ------- | ------------------------------- | ------------------------------------ |
| criterion         | `DecisionTreeClassifier__criterion`         | str   | 'gini'  | ['gini', 'entropy', 'log_loss'] | Function to measure quality of split |
| splitter          | `DecisionTreeClassifier__splitter`          | str   | 'best'  | ['best', 'random']              | Strategy to choose split             |
| max_depth         | `DecisionTreeClassifier__max_depth`         | int   | None    | [1, 1000]                       | Maximum depth of tree                |
| min_samples_split | `DecisionTreeClassifier__min_samples_split` | float | 2       | [0.01, 1.0]                     | Minimum samples for split            |
| min_samples_leaf  | `DecisionTreeClassifier__min_samples_leaf`  | float | 1       | [0.01, 1.0]                     | Minimum samples at leaf              |

##### GaussianProcessClassifier

**Class**: `GaussianProcessClassifier`
**Module**: `autoqml.search_space.classification.classic.gaussian_process_classifier`

| Parameter   | Config Key                               | Type | Default       | Range/Choices                   | Description             |
| ----------- | ---------------------------------------- | ---- | ------------- | ------------------------------- | ----------------------- |
| kernel      | `GaussianProcessClassifier__kernel`      | str  | 'RBF'         | ['RBF', 'DotProduct', 'Matern'] | Kernel type             |
| warm_start  | `GaussianProcessClassifier__warm_start`  | bool | False         | [True, False]                   | Reuse previous solution |
| multi_class | `GaussianProcessClassifier__multi_class` | str  | 'one_vs_rest' | ['one_vs_rest', 'one_vs_one']   | Multi-class strategy    |

##### LogisticRegressor

**Class**: `LogisticRegressor`
**Module**: `autoqml.search_space.classification.classic.logistic_regression_classifier`

| Parameter     | Config Key                         | Type  | Default | Range/Choices                    | Description                     |
| ------------- | ---------------------------------- | ----- | ------- | -------------------------------- | ------------------------------- |
| penalty       | `LogisticRegressor__penalty`       | str   | 'l2'    | [None, 'l2', 'l1', 'elasticnet'] | Regularization type             |
| dual          | `LogisticRegressor__dual`          | bool  | False   | [True, False]                    | Dual formulation                |
| C             | `LogisticRegressor__C`             | float | 1.0     | [0.01, 10000]                    | Inverse regularization strength |
| fit_intercept | `LogisticRegressor__fit_intercept` | bool  | True    | [True, False]                    | Add intercept term              |
| max_iter      | `LogisticRegressor__max_iter`      | int   | 100     | [10, 10000]                      | Maximum iterations              |

##### Perceptron

**Class**: `Perceptron`
**Module**: `autoqml.search_space.classification.classic.perceptron`

| Parameter | Config Key            | Type  | Default | Range/Choices      | Description             |
| --------- | --------------------- | ----- | ------- | ------------------ | ----------------------- |
| penalty   | `Perceptron__penalty` | str   | None    | [None, 'l2', 'l1'] | Regularization type     |
| alpha     | `Perceptron__alpha`   | float | 0.001   | [0.03125, 32768]   | Regularization strength |

##### RandomForestClassifier

**Class**: `RandomForestClassifier`
**Module**: `autoqml.search_space.classification.classic.random_forest_classifier`

| Parameter         | Config Key                                  | Type  | Default | Range/Choices                   | Description               |
| ----------------- | ------------------------------------------- | ----- | ------- | ------------------------------- | ------------------------- |
| criterion         | `RandomForestClassifier__criterion`         | str   | 'gini'  | ['gini', 'entropy', 'log_loss'] | Splitting criterion       |
| max_depth         | `RandomForestClassifier__max_depth`         | int   | None    | [1, 1000]                       | Maximum tree depth        |
| min_samples_split | `RandomForestClassifier__min_samples_split` | float | 2       | [0.01, 1.0]                     | Minimum samples for split |
| min_samples_leaf  | `RandomForestClassifier__min_samples_leaf`  | float | 1       | [0.01, 1.0]                     | Minimum samples in leaf   |

##### RidgeClassifier

**Class**: `RidgeClassifier`
**Module**: `autoqml.search_space.classification.classic.ridge_classifier`

| Parameter     | Config Key                       | Type  | Default | Range/Choices    | Description             |
| ------------- | -------------------------------- | ----- | ------- | ---------------- | ----------------------- |
| alpha         | `RidgeClassifier__alpha`         | float | 1.0     | [0.03125, 32768] | Regularization strength |
| fit_intercept | `RidgeClassifier__fit_intercept` | bool  | True    | [True, False]    | Add intercept term      |

##### SVC

**Class**: `SVC`
**Module**: `autoqml.search_space.classification.classic.svc`

| Parameter | Config Key    | Type  | Default | Range/Choices              | Description              |
| --------- | ------------- | ----- | ------- | -------------------------- | ------------------------ |
| kernel    | `SVC__kernel` | str   | 'rbf'   | ['rbf', 'poly', 'sigmoid'] | Kernel type              |
| C         | `SVC__C`      | float | 0.1     | [0.03125, 32768]           | Regularization parameter |

#### Quantum Classifiers

##### QGPC

**Class**: `QGPC`
**Module**: `autoqml.search_space.classification.quantum.qgpc`

| Parameter  | Config Key         | Type | Default | Range/Choices | Description      |
| ---------- | ------------------ | ---- | ------- | ------------- | ---------------- |
| num_qubits | `QGPC__num_qubits` | int  | 3       | [1, 2, 4, 8]  | Number of qubits |

**Encoding Circuit Parameters**:
| Parameter | Config Key | Type | Default | Range/Choices | Description |
| --------- | ---------- | ---- | ------- | ------------- | ----------- |
| encoding_circuit | `QGPC__encoding_circuit` | str | 'chebyshev_pqc' | - | Circuit type |
| num_layers | `QGPC__num_layers` | int | 2 | - | Number of layers |
| chebyshev_alpha | `QGPC__chebyshev_alpha` | float | 1.0 | - | Chebyshev parameter |
| num_chebyshev | `QGPC__num_chebyshev` | int | None | - | Chebyshev terms |

**Quantum Kernel Parameters**:
| Parameter | Config Key | Type | Default | Range/Choices | Description |
| --------- | ---------- | ---- | ------- | ------------- | ----------- |
| quantum_kernel | `QGPC__quantum_kernel` | str | 'projected_quantum_kernel' | - | Kernel type |
| measurement | `QGPC__measurement` | str | 'X' | - | Measurement basis |
| outer_kernel | `QGPC__outer_kernel` | str | 'gaussian' | - | Classical kernel |
| parameter_seed | `QGPC__parameter_seed` | int | 0 | [0, 2^32-1] | Random seed |

##### QNNClassifier

**Class**: `QNNClassifier`
**Module**: `autoqml.search_space.classification.quantum.qnnc`

| Parameter      | Config Key                      | Type  | Default | Range/Choices                        | Description      |
| -------------- | ------------------------------- | ----- | ------- | ------------------------------------ | ---------------- |
| num_qubits     | `QNNClassifier__num_qubits`     | int   | 3       | [1, 2, 4, 8]                         | Number of qubits |
| learning_rate  | `QNNClassifier__learning_rate`  | float | 0.1     | [0.0001, 0.1]                        | Learning rate    |
| batch_size     | `QNNClassifier__batch_size`     | int   | 10      | [1, 100]                             | Batch size       |
| epochs         | `QNNClassifier__epochs`         | int   | 10      | [5, 100]                             | Training epochs  |
| shuffle        | `QNNClassifier__shuffle`        | bool  | True    | [True, False]                        | Shuffle data     |
| variance       | `QNNClassifier__variance`       | float | None    | [None, 0.001, 0.005, 0.0001, 0.0005] | Noise variance   |
| parameter_seed | `QNNClassifier__parameter_seed` | int   | 0       | [0, 2^32-1]                          | Random seed      |

**Encoding Circuit Parameters**:
| Parameter | Config Key | Type | Default | Range/Choices | Description |
| --------- | ---------- | ---- | ------- | ------------- | ----------- |
| encoding_circuit | `QNNClassifier__encoding_circuit` | str | 'chebyshev_pqc' | - | Circuit type |
| num_layers | `QNNClassifier__num_layers` | int | 2 | - | Number of layers |
| chebyshev_alpha | `QNNClassifier__chebyshev_alpha` | float | 1.0 | - | Chebyshev parameter |
| num_chebyshev | `QNNClassifier__num_chebyshev` | int | None | - | Chebyshev terms |

**Observable Parameters**:
| Parameter | Config Key | Type | Default | Range/Choices | Description |
| --------- | ---------- | ---- | ------- | ------------- | ----------- |
| observable | `QNNClassifier__observable` | str | 'single_pauli_x' | - | Observable type |
| observable_qubit | `QNNClassifier__observable_qubit` | int | 0 | - | Measured qubit |

##### QRCClassifier

**Class**: `QRCClassifier`
**Module**: `autoqml.search_space.classification.quantum.qrcc`

| Parameter      | Config Key                      | Type | Default         | Range/Choices                      | Description         |
| -------------- | ------------------------------- | ---- | --------------- | ---------------------------------- | ------------------- |
| num_qubits     | `QRCClassifier__num_qubits`     | int  | 3               | [1, 2, 4, 8]                       | Number of qubits    |
| ml_model       | `QRCClassifier__ml_model`       | str  | 'linear'        | ['mlp', 'linear', 'kernel']        | Classical model     |
| operators      | `QRCClassifier__operators`      | str  | 'random_paulis' | ['random_paulis', 'single_paulis'] | Operator type       |
| num_operators  | `QRCClassifier__num_operators`  | int  | 100             | [1, 10000]                         | Number of operators |
| operator_seed  | `QRCClassifier__operator_seed`  | int  | 0               | [0, 2^32-1]                        | Operator seed       |
| parameter_seed | `QRCClassifier__parameter_seed` | int  | 0               | [0, 2^32-1]                        | Parameter seed      |

**Encoding Circuit Parameters**:
| Parameter | Config Key | Type | Default | Range/Choices | Description |
| --------- | ---------- | ---- | ------- | ------------- | ----------- |
| encoding_circuit | `QRCClassifier__encoding_circuit` | str | 'chebyshev_pqc' | - | Circuit type |
| num_layers | `QRCClassifier__num_layers` | int | 2 | - | Number of layers |
| chebyshev_alpha | `QRCClassifier__chebyshev_alpha` | float | 1.0 | - | Chebyshev parameter |
| num_chebyshev | `QRCClassifier__num_chebyshev` | int | None | - | Chebyshev terms |

##### QSVC

**Class**: `QSVC`
**Module**: `autoqml.search_space.classification.quantum.qsvc`

| Parameter        | Config Key               | Type            | Default                    | Range/Choices                     | Description                            |
| ---------------- | ------------------------ | --------------- | -------------------------- | --------------------------------- | -------------------------------------- |
| encoding_circuit | `QSVC__encoding_circuit` | str             | 'chebyshev_pqc'            | ['chebyshev_pqc']                 | Type of quantum encoding circuit       |
| num_qubits       | `QSVC__num_qubits`       | int             | 3                          | [1, 2, 4, 8]                      | Number of qubits in circuit            |
| num_layers       | `QSVC__num_layers`       | int             | 2                          | [1, 100]                          | Number of circuit layers               |
| chebyshev_alpha  | `QSVC__chebyshev_alpha`  | float           | 1.0                        | [0.1, 10.0]                       | Alpha parameter for Chebyshev          |
| num_chebyshev    | `QSVC__num_chebyshev`    | int             | None                       | [1, 100]                          | Number of Chebyshev polynomials        |
| quantum_kernel   | `QSVC__quantum_kernel`   | str             | 'projected_quantum_kernel' | ['projected_quantum_kernel']      | Type of quantum kernel                 |
| measurement      | `QSVC__measurement`      | Union[str,list] | 'X'                        | ['X', 'Y', 'Z'] or ObservableBase | Measurement basis or custom observable |
| outer_kernel     | `QSVC__outer_kernel`     | str             | 'gaussian'                 | ['gaussian', 'linear']            | Classical kernel for outer product     |
| parameter_seed   | `QSVC__parameter_seed`   | int             | 0                          | [0, 2^32-1]                       | Random seed for parameters             |
| C                | `QSVC__C`                | float           | 100                        | [0.03125, 32768]                  | Regularization parameter               |

## Regression Components

Components for building regression models, including both classical and quantum regressors.

### Classical Regressors

#### DecisionTreeRegressor

**Class**: `DecisionTreeRegressor`
**Module**: `autoqml.search_space.regression.classic.decision_tree_regressor`

| Parameter         | Config Key                                 | Type  | Default         | Range/Choices                                                  | Description                        |
| ----------------- | ------------------------------------------ | ----- | --------------- | -------------------------------------------------------------- | ---------------------------------- |
| criterion         | `DecisionTreeRegressor__criterion`         | str   | 'squared_error' | ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] | Split quality criterion            |
| splitter          | `DecisionTreeRegressor__splitter`          | str   | 'best'          | ['best', 'random']                                             | Strategy to choose split           |
| max_depth         | `DecisionTreeRegressor__max_depth`         | int   | None            | [1, 1000]                                                      | Maximum depth of tree              |
| min_samples_split | `DecisionTreeRegressor__min_samples_split` | float | 2               | [0.01, 1.0]                                                    | Min samples to split internal node |
| min_samples_leaf  | `DecisionTreeRegressor__min_samples_leaf`  | float | 1               | [0.01, 1.0]                                                    | Min samples required at leaf node  |

#### GaussianProcessRegressor

**Class**: `GaussianProcessRegressor`
**Module**: `autoqml.search_space.regression.classic.gaussian_process_regressor`

| Parameter   | Config Key                              | Type  | Default | Range/Choices                   | Description                              |
| ----------- | --------------------------------------- | ----- | ------- | ------------------------------- | ---------------------------------------- |
| kernel      | `GaussianProcessRegressor__kernel`      | str   | 'RBF'   | ['RBF', 'DotProduct', 'Matern'] | Kernel type for the GP                   |
| alpha       | `GaussianProcessRegressor__alpha`       | float | 0.1     | [0.03125, 32768]                | Value added to diagonal of kernel matrix |
| normalize_y | `GaussianProcessRegressor__normalize_y` | bool  | False   | [True, False]                   | Whether to normalize target values       |

#### KernelRidge

**Class**: `KernelRidge`
**Module**: `autoqml.search_space.regression.classic.kernel_ridge`

| Parameter | Config Key            | Type  | Default | Range/Choices              | Description                          |
| --------- | --------------------- | ----- | ------- | -------------------------- | ------------------------------------ |
| alpha     | `KernelRidge__alpha`  | float | 0.1     | [0.03125, 32768]           | Regularization strength              |
| kernel    | `KernelRidge__kernel` | str   | 'rbf'   | ['rbf', 'poly', 'sigmoid'] | Kernel type                          |
| gamma     | `KernelRidge__gamma`  | float | 1       | [0.0001, 10000]            | Kernel coefficient for rbf           |
| degree    | `KernelRidge__degree` | int   | 3       | [1, 10]                    | Polynomial degree (poly kernel only) |
| coef0     | `KernelRidge__coef0`  | float | 1       | [0.03125, 32768]           | Independent term (poly/sigmoid only) |

#### LinearRegressor

**Class**: `LinearRegressor`
**Module**: `autoqml.search_space.regression.classic.linear_regressor`

| Parameter     | Config Key                       | Type | Default | Range/Choices | Description                 |
| ------------- | -------------------------------- | ---- | ------- | ------------- | --------------------------- |
| fit_intercept | `LinearRegressor__fit_intercept` | bool | True    | [True, False] | Include intercept term      |
| positive      | `LinearRegressor__positive`      | bool | False   | [True, False] | Force positive coefficients |

#### NNRegressor

**Class**: `NNRegressor`
**Module**: `autoqml.search_space.regression.classic.nnr`

| Parameter        | Config Key                      | Type | Default    | Range/Choices                                    | Description              |
| ---------------- | ------------------------------- | ---- | ---------- | ------------------------------------------------ | ------------------------ |
| n_hidden_neurons | `NNRegressor__n_hidden_neurons` | int  | 128        | [32, 512]                                        | Number of hidden neurons |
| act_fn           | `NNRegressor__act_fn`           | str  | 'GELU'     | ['ReLU', 'LeakyReLU', 'GELU', 'Sigmoid', 'Tanh'] | Activation function      |
| loss_fn          | `NNRegressor__loss_fn`          | str  | 'mse_loss' | ['mse_loss', 'mae_loss']                         | Loss function            |
| max_epochs       | `NNRegressor__max_epochs`       | int  | 200        | [32, 512]                                        | Maximum training epochs  |

**Note**: This is a PyTorch Lightning-based neural network regressor with a 3-layer architecture. The hidden layer sizes are [n_hidden_neurons, n_hidden_neurons/2, 1].

#### RandomForestRegressor

**Class**: `RandomForestRegressor`
**Module**: `autoqml.search_space.regression.classic.random_forest_regressor`

| Parameter         | Config Key                                 | Type  | Default         | Range/Choices                                                  | Description               |
| ----------------- | ------------------------------------------ | ----- | --------------- | -------------------------------------------------------------- | ------------------------- |
| n_estimators      | `RandomForestRegressor__n_estimators`      | int   | 100             | [10, 10000]                                                    | Number of trees in forest |
| criterion         | `RandomForestRegressor__criterion`         | str   | 'squared_error' | ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] | Splitting criterion       |
| max_depth         | `RandomForestRegressor__max_depth`         | int   | None            | [1, 1000]                                                      | Maximum depth of trees    |
| min_samples_split | `RandomForestRegressor__min_samples_split` | float | 2               | [0.01, 1.0]                                                    | Minimum samples for split |
| min_samples_leaf  | `RandomForestRegressor__min_samples_leaf`  | float | 1               | [0.01, 1.0]                                                    | Minimum samples in leaf   |

#### SVR

**Class**: `SVR`
**Module**: `autoqml.search_space.regression.classic.svr`

| Parameter | Config Key | Type  | Default | Range/Choices    | Description              |
| --------- | ---------- | ----- | ------- | ---------------- | ------------------------ |
| C         | `SVR__C`   | float | 0.1     | [0.03125, 32768] | Regularization parameter |

### Quantum Regressors

#### QNNRegressor

**Class**: `QNNRegressor`
**Module**: `autoqml.search_space.regression.quantum.qnnr`

| Parameter        | Config Key                       | Type  | Default          | Range/Choices                                          | Description                     |
| ---------------- | -------------------------------- | ----- | ---------------- | ------------------------------------------------------ | ------------------------------- |
| encoding_circuit | `QNNRegressor__encoding_circuit` | str   | 'chebyshev_pqc'  | ['chebyshev_pqc']                                      | Circuit type for encoding       |
| num_qubits       | `QNNRegressor__num_qubits`       | int   | 3                | [1, 2, 4, 8]                                           | Number of qubits                |
| num_layers       | `QNNRegressor__num_layers`       | int   | 2                | [1, inf)                                               | Number of circuit layers        |
| chebyshev_alpha  | `QNNRegressor__chebyshev_alpha`  | float | 1.0              | (0, inf)                                               | Chebyshev polynomial parameter  |
| num_chebyshev    | `QNNRegressor__num_chebyshev`    | int   | None             | [1, 100]                                               | Number of Chebyshev polynomials |
| observable       | `QNNRegressor__observable`       | str   | 'single_pauli_x' | ['single_pauli_x', 'single_pauli_y', 'single_pauli_z'] | Observable for measurement      |
| observable_qubit | `QNNRegressor__observable_qubit` | int   | 0                | [0, num_qubits-1]                                      | Qubit to measure                |
| learning_rate    | `QNNRegressor__learning_rate`    | float | 0.1              | [0.0001, 0.1]                                          | Learning rate for optimization  |
| epochs           | `QNNRegressor__epochs`           | int   | 10               | [5, 100]                                               | Number of training epochs       |
| shuffle          | `QNNRegressor__shuffle`          | bool  | True             | [True, False]                                          | Shuffle data                    |
| variance         | `QNNRegressor__variance`         | float | None             | [None, 0.001, 0.005, 0.0001, 0.0005]                   | Noise variance                  |
| batch_size       | `QNNRegressor__batch_size`       | int   | 10               | [1, 100]                                               | Training batch size             |
| parameter_seed   | `QNNRegressor__parameter_seed`   | int   | 0                | [0, 2^32-1]                                            | Random seed for parameters      |

#### QRCRegressor

**Class**: `QRCRegressor`
**Module**: `autoqml.search_space.regression.quantum.qrc_regressor`

| Parameter        | Config Key                       | Type  | Default         | Range/Choices      | Description                         |
| ---------------- | -------------------------------- | ----- | --------------- | ------------------ | ----------------------------------- |
| encoding_circuit | `QRCRegressor__encoding_circuit` | str   | 'chebyshev_pqc' | ['chebyshev_pqc']  | Circuit type for encoding           |
| num_qubits       | `QRCRegressor__num_qubits`       | int   | None            | [1, inf)           | Number of qubits                    |
| num_layers       | `QRCRegressor__num_layers`       | int   | 2               | [1, inf)           | Number of circuit layers            |
| chebyshev_alpha  | `QRCRegressor__chebyshev_alpha`  | float | 1.0             | (0, inf)           | Chebyshev polynomial parameter      |
| num_chebyshev    | `QRCRegressor__num_chebyshev`    | int   | None            | [1, 100]           | Number of Chebyshev polynomials     |
| ml_model         | `QRCRegressor__ml_model`         | str   | 'ridge'         | ['ridge', 'lasso'] | Classical ML model for readout      |
| operators        | `QRCRegressor__operators`        | list  | None            | -                  | Custom measurement operators        |
| num_operators    | `QRCRegressor__num_operators`    | int   | None            | [1, inf)           | Number of measurement operators     |
| operator_seed    | `QRCRegressor__operator_seed`    | int   | 0               | [0, 2^32-1]        | Random seed for operator generation |
| parameter_seed   | `QRCRegressor__parameter_seed`   | int   | 0               | [0, 2^32-1]        | Random seed for parameters          |

#### QSVR

**Class**: `QSVR`
**Module**: `autoqml.search_space.regression.quantum.qsvr`

| Parameter        | Config Key               | Type            | Default                    | Range/Choices                                           | Description                            |
| ---------------- | ------------------------ | --------------- | -------------------------- | ------------------------------------------------------- | -------------------------------------- |
| encoding_circuit | `QSVR__encoding_circuit` | str             | 'chebyshev_pqc'            | ['chebyshev_pqc']                                       | Circuit type for encoding              |
| num_qubits       | `QSVR__num_qubits`       | int             | None                       | [1, inf)                                                | Number of qubits                       |
| num_layers       | `QSVR__num_layers`       | int             | 2                          | [1, inf)                                                | Number of circuit layers               |
| chebyshev_alpha  | `QSVR__chebyshev_alpha`  | float           | 1.0                        | (0, inf)                                                | Chebyshev polynomial parameter         |
| num_chebyshev    | `QSVR__num_chebyshev`    | int             | None                       | [1, 100]                                                | Number of Chebyshev polynomials        |
| quantum_kernel   | `QSVR__quantum_kernel`   | str             | 'projected_quantum_kernel' | ['projected_quantum_kernel', 'fidelity_quantum_kernel'] | Type of quantum kernel                 |
| measurement      | `QSVR__measurement`      | Union[str,list] | 'X'                        | ['X', 'Y', 'Z'] or ObservableBase                       | Measurement basis or custom observable |
| outer_kernel     | `QSVR__outer_kernel`     | str             | 'gaussian'                 | ['gaussian', 'linear']                                  | Classical kernel for outer product     |
| parameter_seed   | `QSVR__parameter_seed`   | int             | 0                          | [0, 2^32-1]                                             | Random seed for parameters             |
| C                | `QSVR__C`                | float           | 100                        | [0.03125, 32768]                                        | Regularization parameter               |

#### QKRR

**Class**: `QKRR`
**Module**: `autoqml.search_space.regression.quantum.qkrr`

| Parameter        | Config Key               | Type            | Default                    | Range/Choices                                           | Description                            |
| ---------------- | ------------------------ | --------------- | -------------------------- | ------------------------------------------------------- | -------------------------------------- |
| encoding_circuit | `QKRR__encoding_circuit` | str             | 'chebyshev_pqc'            | ['chebyshev_pqc']                                       | Circuit type for encoding              |
| num_qubits       | `QKRR__num_qubits`       | int             | None                       | [1, inf)                                                | Number of qubits                       |
| num_layers       | `QKRR__num_layers`       | int             | 2                          | [1, inf)                                                | Number of circuit layers               |
| chebyshev_alpha  | `QKRR__chebyshev_alpha`  | float           | 1.0                        | (0, inf)                                                | Chebyshev polynomial parameter         |
| num_chebyshev    | `QKRR__num_chebyshev`    | int             | None                       | [1, 100]                                                | Number of Chebyshev polynomials        |
| quantum_kernel   | `QKRR__quantum_kernel`   | str             | 'projected_quantum_kernel' | ['projected_quantum_kernel', 'fidelity_quantum_kernel'] | Type of quantum kernel                 |
| measurement      | `QKRR__measurement`      | Union[str,list] | 'X'                        | ['X', 'Y', 'Z'] or ObservableBase                       | Measurement basis or custom observable |
| outer_kernel     | `QKRR__outer_kernel`     | str             | 'gaussian'                 | ['gaussian', 'linear']                                  | Classical kernel for outer product     |
| parameter_seed   | `QKRR__parameter_seed`   | int             | 0                          | [0, 2^32-1]                                             | Random seed for parameters             |
| C                | `QKRR__C`                | float           | 100                        | [0.03125, 32768]                                        | Regularization parameter               |

#### QGPR

**Class**: `QGPR`
**Module**: `autoqml.search_space.regression.quantum.qgpr`

| Parameter        | Config Key               | Type            | Default                    | Range/Choices                                           | Description                            |
| ---------------- | ------------------------ | --------------- | -------------------------- | ------------------------------------------------------- | -------------------------------------- |
| encoding_circuit | `QGPR__encoding_circuit` | str             | 'chebyshev_pqc'            | ['chebyshev_pqc']                                       | Circuit type for encoding              |
| num_qubits       | `QGPR__num_qubits`       | int             | None                       | [1, inf)                                                | Number of qubits                       |
| num_layers       | `QGPR__num_layers`       | int             | 2                          | [1, inf)                                                | Number of circuit layers               |
| chebyshev_alpha  | `QGPR__chebyshev_alpha`  | float           | 1.0                        | (0, inf)                                                | Chebyshev polynomial parameter         |
| num_chebyshev    | `QGPR__num_chebyshev`    | int             | None                       | [1, 100]                                                | Number of Chebyshev polynomials        |
| quantum_kernel   | `QGPR__quantum_kernel`   | str             | 'projected_quantum_kernel' | ['projected_quantum_kernel', 'fidelity_quantum_kernel'] | Type of quantum kernel                 |
| measurement      | `QGPR__measurement`      | Union[str,list] | 'X'                        | ['X', 'Y', 'Z'] or ObservableBase                       | Measurement basis or custom observable |
| outer_kernel     | `QGPR__outer_kernel`     | str             | 'gaussian'                 | ['gaussian', 'linear']                                  | Classical kernel for outer product     |
| parameter_seed   | `QGPR__parameter_seed`   | int             | 0                          | [0, 2^32-1]                                             | Random seed for parameters             |

## Configuration Examples

### Basic Configurations

#### Single Algorithm Configuration

```python
config = {
    'autoqml.search_space.regression.RegressionChoice__choice': 'svr',
    'autoqml.search_space.regression.svr.SVR__kernel': 'rbf',
    'autoqml.search_space.regression.svr.SVR__C': 1.0,
}
```

#### Multiple Component Configuration

```python
config = {
    'autoqml.search_space.regression.RegressionChoice__choice': 'svr',
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice': 'standard_scaling',
    'autoqml.search_space.preprocessing.encoding.EncoderChoice__choice': 'one-hot',
    'autoqml.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories': 17,
}
```

### Real-World Examples

#### Time Series Classification

```python
config = {
    'autoqml.search_space.classification.ClassificationChoice__choice': 'random_forest_classifier',
    'autoqml.search_space.data_loading.timeseries.TabularizeTimeSeries__tile_size_x': 7,
    'autoqml.search_space.data_loading.timeseries.TabularizeTimeSeries__tile_size_y': 30,
}
```

#### Tabular Regression

```python
config = {
    'autoqml.search_space.regression.RegressionChoice__choice': 'qsvr',
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice': 'standard_scaling',
    'autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice': 'pca',
    'autoqml.search_space.preprocessing.dim_reduction.pca.PCA__n_components': 10,
}
```

### Common Patterns

#### Default Configuration Override

```python
# Only override specific parameters
config = {
    'autoqml.search_space.regression.RegressionChoice__choice': 'svr',
    'autoqml.search_space.regression.svr.SVR__C': 2.0,  # Override default C value
}
```

#### Component Selection

```python
# Select components for each pipeline stage
config = {
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice': 'standard_scaling',
    'autoqml.search_space.preprocessing.encoding.EncoderChoice__choice': 'one-hot',
    'autoqml.search_space.regression.RegressionChoice__choice': 'svr',
}
```

## Error Handling

### Common Configuration Errors

1. **Invalid Component Choice**

   ```python
   # Error: 'svm' is not a valid choice
   config = {
       'autoqml.search_space.regression.RegressionChoice__choice': 'svm',  # Should be 'svr'
   }
   ```

2. **Invalid Parameter Value**

   ```python
   # Error: negative C value
   config = {
       'autoqml.search_space.regression.svr.SVR__C': -1.0,  # Must be positive
   }
   ```

3. **Missing Required Parameters**
   ```python
   # Error: missing required choice parameter
   config = {
       'autoqml.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories': 17,
       # Missing: 'autoqml.search_space.preprocessing.encoding.EncoderChoice__choice'
   }
   ```

### How to Fix Common Errors

1. Check the documentation for valid component choices
2. Verify parameter value ranges
3. Ensure all required parameters are specified
4. Use the correct configuration key format

## Best Practices

1. **Start Simple**

   - Begin with default configurations
   - Modify one component at a time
   - Test changes incrementally

2. **Configuration Organization**

   - Group related parameters together
   - Use clear variable names
   - Comment complex configurations

3. **Performance Optimization**

   - Configure preprocessing for your data size
   - Choose appropriate algorithm complexity
   - Balance pipeline complexity with data size

4. **Error Prevention**
   - Validate configurations before running
   - Keep track of successful configurations
   - Document configuration changes

## Workflow Integration Guide

### Basic Workflow Integration

```python
from autoqml import AutoQMLTabularRegression
from autoqml import AutoQMLFitCommand

# Create configuration
config = {
    'autoqml.search_space.regression.RegressionChoice__choice': 'svr',
}

# Initialize AutoQML
automl = AutoQMLTabularRegression()

# Create fit command
cmd = AutoQMLFitCommand(
    X_train,
    y_train,
    timedelta(minutes=5),
    configuration=config,
)

# Fit the model
automl.fit(cmd)
```

### Advanced Workflow Integration

1. **Custom Pipeline Components**

   - Implement the TunableMixin interface
   - Register components with AutoQML
   - Add custom configuration parameters

2. **Automated Parameter Tuning**

   - Use the tuning interface
   - Define parameter search spaces
   - Implement custom optimization metrics

3. **Production Deployment**
   - Save and load configurations
   - Version control your configs
   - Monitor model performance

## Performance Optimization

### Configuration Tips

1. **Data Loading**

   - Adjust batch sizes for your memory constraints
   - Use appropriate data types
   - Configure caching behavior

2. **Preprocessing**

   - Choose efficient encoding methods
   - Configure dimensionality reduction
   - Balance precision and speed

3. **Model Selection**
   - Match model complexity to data size
   - Configure training time limits
   - Use appropriate convergence criteria

### Resource Usage

1. **Memory Optimization**

   - Configure batch processing
   - Use sparse data structures
   - Manage feature dimensionality

2. **Computation Optimization**
   - Set appropriate iteration limits
   - Configure parallel processing
   - Use hardware acceleration

## Downsampling

One way to optimize the size of the data used for training is to downsample the data. This can be done using the `Resampling` transformer of AutoQML.

### Resampling

**Class**: `Resampling`
**Module**: `autoqml.search_space.preprocessing.downsampling.resampling`

| Parameter | Config Key              | Type | Default | Range/Choices  | Description                            |
| --------- | ----------------------- | ---- | ------- | -------------- | -------------------------------------- |
| stratify  | `Resampling__stratify`  | bool | False   | [True, False]  | Whether to preserve class distribution |

## NoOp

**Class**: `NoOp`
**Module**: `autoqml.search_space.preprocessing.no_op`

A pass-through transformer that returns the input data unchanged. Useful as a placeholder, for example in preprocessing pipelines.

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/AutoQML/autoqml/issues) for similar problems
2. Review the API documentation for correct usage
3. Create a new issue with:
   - Full error traceback
   - Minimal reproducible example
   - System information
   - Package versions
