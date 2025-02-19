# Examples

## Basic Usage Examples

### Automatic Time Series Classification Pipeline Configuration

Note that fully automatic pipelines &mdash; `custom_config = {}` &mdash; require a significantly longer amount of time to produce a good pipeline.

```python
from autoqml_lib.automl import AutoQMLTimeSeriesClassification
from autoqml_lib.messages import AutoQMLFitCommand
from datetime import timedelta
import numpy as np

# Create sample data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
pipeline = AutoQMLTimeSeriesClassification()
cmd = AutoQMLFitCommand(
    X_train, 
    y_train, 
    timedelta(seconds=300),
    custom_config = {})
pipeline.fit(cmd)

# Make predictions
predictions = pipeline.predict(X_test)
```

### Custom Pipeline Configuration

```python
custom_config = {
    'autoqml_lib.search_space.classification.ClassificationChoice__choice': 'qsvc',
    'autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice': 'min_max_scaling',
    'autoqml_lib.search_space.classification.quantum.qsvc.QSVC__num_layers': 4,
    'autoqml_lib.search_space.classification.quantum.qsvc.QSVC__num_qubits': 3,
    'autoqml_lib.search_space.classification.quantum.qsvc.QSVC__C': 100,
}

cmd = AutoQMLFitCommand(
    X_train,
    y_train,
    timedelta(seconds=300),
    configuration=custom_config
)
```

## Advanced Use Cases

### KEB Use Case

The KEB use case demonstrates time series classification:

1. Data Loading:
```python
# Load your time series data
X, y = load_keb_data()
```

2. Training:
```python
pipeline = AutoQMLTimeSeriesClassification()
cmd = AutoQMLFitCommand(X_train, y_train, timedelta(seconds=300))
pipeline.fit(cmd)
```

3. Evaluation:
```python
from sklearn.metrics import classification_report

predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))
```

### Zeppelin Use Case

The Zeppelin use case shows a regression pipeline with autoencoder for data reduction:

1. Setup:
```python
from autoqml_lib.automl import AutoQMLTabularRegression

pipeline = AutoQMLTabularRegression()
```

2. Training with autoencoder:
```python
config = {
    'autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice': 'autoencoder'
}

cmd = AutoQMLFitCommand(
    X_train,
    y_train,
    timedelta(seconds=600),
    configuration=config
)
pipeline.fit(cmd)
```

For more use cases and complete examples for the above use cases, check the `use_cases` directory in the project repository.
