# Examples

## Basic Usage Examples

### Automatic Time Series Classification Pipeline Configuration

Note that fully automatic pipelines &mdash; `custom_config = {}` &mdash; require a significantly longer amount of time to produce a good pipeline.

```python
from autoqml import AutoQMLTimeSeriesClassification
from autoqml import AutoQMLFitCommand
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
    'autoqml.search_space.classification.ClassificationChoice__choice': 'qsvc',
    'autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice': 'min_max_scaling',
    'autoqml.search_space.classification.quantum.qsvc.QSVC__num_layers': 4,
    'autoqml.search_space.classification.quantum.qsvc.QSVC__num_qubits': 3,
    'autoqml.search_space.classification.quantum.qsvc.QSVC__C': 100,
}

cmd = AutoQMLFitCommand(
    X_train,
    y_train,
    timedelta(seconds=300),
    configuration=custom_config
)
```
