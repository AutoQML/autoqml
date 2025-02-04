from datetime import timedelta
import ray
import sklearn
import sklearn.datasets
from autoqml.automl import AutoQMLTabularClassification, AutoQMLTabularRegression
from autoqml.messages import AutoQMLFitCommand
from autoqml.search_space.classification.classic.gaussian_process_classifier import GaussianProcessClassifier
from autoqml.search_space.classification.classic.decision_tree_classifier import DecisionTreeClassifier
from autoqml.search_space.classification.classic.perceptron import Perceptron
from autoqml.search_space.classification.classic.random_forest_classifier import RandomForestClassifier
from autoqml.search_space.classification.classic.ridge_classifier import RidgeClassifier
from autoqml.search_space.classification.classic.svc import SVC
from autoqml.search_space.classification.classic.logistic_regression_classifier import LogisticRegressor

from autoqml.search_space.regression.classic.gaussian_process_regressor import GaussianProcessRegressor
from autoqml.search_space.regression.classic.decision_tree_regressor import DecisionTreeRegressor
from autoqml.search_space.regression.classic.kernel_ridge import KernelRidge
from autoqml.search_space.regression.classic.linear_regressor import LinearRegressor
from autoqml.search_space.regression.classic.random_forest_regressor import RandomForestRegressor
from autoqml.search_space.regression.classic.svr import SVR

from autoqml.search_space.regression.quantum.qsvr import QSVR
from autoqml.search_space.regression.quantum.qgpr import QGPR
from autoqml.search_space.regression.quantum.qkrr import QKRR
from autoqml.search_space.regression.quantum.qnnr import QNNRegressor
from autoqml.search_space.regression.quantum.qrcr import QRCRegressor
from autoqml.search_space.classification.quantum.qgpc import QGPC
from autoqml.search_space.classification.quantum.qsvc import QSVC
from autoqml.search_space.classification.quantum.qnnc import QNNClassifier
from autoqml.search_space.classification.quantum.qrcc import QRCClassifier
import pytest


def get_fullname(cls: type) -> str:
    return cls.__module__ + "." + cls.__name__ + "__"

def generate_gaussian_process_classifier_configurations(cls: type, build_prefix: callable) -> list[dict]:
    
    kernel = ['RBF', 'DotProduct', 'Matern']
    warm_start = [True, False]
    multi_class = ['one_vs_rest', 'one_vs_one']
    
    configurations = []
    
    for k in kernel:
        for ws in warm_start:
            for mc in multi_class:
                configurations.append({
                    build_prefix(cls) + "kernel": k,
                    build_prefix(cls) + "warm_start": ws,
                    build_prefix(cls) + "multi_class": mc
                })
                
    return configurations

def generate_gaussian_process_regressor_configurations(cls: type, build_prefix: callable) -> list[dict]:
    
    kernel = ['RBF', 'DotProduct', 'Matern']
    normalize_y = [True, False]
    
    configurations = []
    
    for k in kernel:
        for n_y in normalize_y:

            configurations.append({
                build_prefix(cls) + "kernel": k,
                build_prefix(cls) + "normalize_y ": n_y
            })
                
    return configurations

def generate_decision_tree_classifier_configurations(cls: type, build_prefix: callable) -> list[dict]:

    criterion = ['gini', 'entropy', 'log_loss']
    splitter = ['best', 'random']

    configurations = []

    for c in criterion:
        for s in splitter:
            configurations.append({
                build_prefix(cls) + 'criterion': c,
                build_prefix(cls) + 'splitter': s
            })

    return configurations

def generate_decision_tree_regression_configurations(cls: type, build_prefix: callable) -> list[dict]:

    criterion = ['squared_error', 'absolute_error', 'friedman_mse']
    splitter = ['best', 'random']

    configurations = []
    
    for c in criterion:
        for s in splitter:
            configurations.append({
                build_prefix(cls) + 'criterion': c,
                build_prefix(cls) + 'splitter': s
            })

    return configurations

def generate_perceptron_classifier_configurations(cls: type, build_prefix: callable) -> list[dict]:

    penalty = [None, 'l2', 'l1']

    configurations = []
    
    for p in penalty:
        configurations.append({
            build_prefix(cls) + 'penalty': p
        })

    return configurations

def generate_random_forest_classifier_configurations(cls: type, build_prefix: callable) -> list[dict]:
    criterion = ['gini', 'entropy', 'log_loss']

    configurations = []
    
    for c in criterion:
        configurations.append({
            build_prefix(cls) + 'criterion': c
        })

    return configurations

def generate_random_forest_regression_configurations(cls: type, build_prefix: callable) -> list[dict]:
    criterion = ['squared_error', 'absolute_error', 'friedman_mse']

    configurations = []
    
    for c in criterion:
        configurations.append({
            build_prefix(cls) + 'criterion': c
        })

    return configurations

def generate_ridge_classifier_configurations(cls: type, build_prefix: callable) -> list[dict]:

    fit_intercept = [True, False]

    configurations = []
    
    for fi in fit_intercept:
        configurations.append({
            build_prefix(cls) + 'fit_intercept': fi
        })

    return configurations

def generate_kernel_ridge_configurations(cls: type, build_prefix: callable) -> list[dict]:
    
    kernel = ['rbf', 'poly', 'sigmoid']

    configurations = []
    
    for k in kernel:
        configurations.append({
            build_prefix(cls) + 'kernel': k
        })

    return configurations

def generate_linear_regression_configurations(cls: type, build_prefix: callable) -> list[dict]:

    fit_intercept = [True, False]
    positive = [True, False]

    configurations = []

    for fi in fit_intercept:
        for p in positive:
            configurations.append({
                build_prefix(cls) + 'fit_intercept': fi,
                build_prefix(cls) + 'positive': p
            })

    return configurations

def generate_logistic_regression_configurations(cls: type, build_prefix: callable) -> list[dict]:

    fit_intercept = [True, False]
    penalty = ['l2']
    dual = [False]
    
    configurations = []

    for fi in fit_intercept:
        for p in penalty:
            for d in dual:         
                configurations.append({
                    build_prefix(cls) + 'fit_intercept': fi,
                    build_prefix(cls) + 'penalty': p,
                    build_prefix(cls) + 'dual': d
                })

    return configurations

def generate_svc_configurations(cls: type, build_prefix: callable) -> list[dict]:

    kernel = ['rbf', 'poly', 'sigmoid']

    configurations = []

    for k in kernel:
        configurations.append({
            build_prefix(cls) + 'kernel': k
        })

    return configurations

def generate_svr_configurations(cls: type, build_prefix: callable) -> list[dict]:

    kernel = ['rbf', 'poly', 'sigmoid']

    configurations = []

    for k in kernel:
        configurations.append({
            build_prefix(cls) + 'kernel': k
        })

    return configurations

def generate_quantum_kernel_configurations(
    cls: type, build_prefix: callable
) -> list[dict]:
    measurement = "X"
    outer_kernels = [
        "Gaussian",
        "Matern",
        "ExpSineSquared",
        "RationalQuadratic",
        "DotProduct",
        "PairwiseKernel",
    ]

    configurations = []

    # config for all outer kernels but only one measurement
    for outer_kernel in outer_kernels:
        configurations.append(
            {
                build_prefix(cls) + "quantum_kernel": "projected_quantum_kernel",
                build_prefix(cls) + "measurement": measurement,
                build_prefix(cls) + "outer_kernel": outer_kernel,
                build_prefix(cls) + "parameter_seed": 0,
            }
        )

    configurations.append(
        {
            build_prefix(cls) + "quantum_kernel": "fidelity_quantum_kernel",
            build_prefix(cls) + "parameter_seed": 0,
        }
    )

    return configurations


def generate_encoding_circuit_configurations(
    cls: type, build_prefix: callable
) -> list[dict]:
    encoding_circuits = [
        "chebyshev_pqc",
        "chebyshev_rx",
        "chebyshev_tower",
        "high_dim_encoding_circuit",
        "hubregtsen_encoding_circuit",
        "multi_control_encoding_circuit",
        "param_z_feature_map",
        "yz_cx_encoding_circuit",
    ]
    num_qubits = 4
    num_repetitions = 2
    chebyshev_alpha = 1.0
    num_chebyshev = 2

    configurations = []

    for encoding_circuit in encoding_circuits:
        config = {
            build_prefix(cls) + "encoding_circuit": encoding_circuit,
            build_prefix(cls) + "num_qubits": num_qubits,
            build_prefix(cls) + "num_repetitions": num_repetitions,
        }
        if encoding_circuit in ["chebyshev_pqc", "chebyshev_rx", "chebyshev_tower"]:
            config[build_prefix(cls) + "chebyshev_alpha"] = chebyshev_alpha
        if encoding_circuit == "chebyshev_tower":
            config[build_prefix(cls) + "num_chebyshev"] = num_chebyshev

        configurations.append(config)

    return configurations


def generate_observable_configurations(cls: type, build_prefix: callable) -> list[dict]:
    observables = [
        "single_pauli_x",
        "single_pauli_y",
        "single_pauli_z",
        "summed_paulis_x",
        "summed_paulis_y",
        "summed_paulis_z",
        "summed_paulis_xy",
        "summed_paulis_yz",
        "summed_paulis_zx",
        "summed_paulis_xyz",
        "single_probability",
        "summed_probabilities",
        "ising_hamiltonian",
        "ising_hamiltonian_transverse",
    ]

    configurations = []

    for observable in observables:
        config = {build_prefix(cls) + "observable": observable}
        if observable.startswith("single"):
            config[build_prefix(cls) + "observable_qubit"] = 0
        configurations.append(config)

    return configurations


def generate_full_kernel_configurations(
    cls: type, build_prefix: callable
) -> list[dict]:
    quantum_kernel_configs = generate_quantum_kernel_configurations(cls, build_prefix)
    encoding_circuit_configs = generate_encoding_circuit_configurations(
        cls, build_prefix
    )

    configurations = []

    # config for all quantum kernels but only one encoding circuit
    for quantum_kernel_config in quantum_kernel_configs:
        config = {}
        config.update(quantum_kernel_config)
        config.update(encoding_circuit_configs[0])
        if cls.__name__ not in ["QGPR", "QGPC"]:
            config.update({build_prefix(cls) + "C": 100})
        configurations.append(config)

    return configurations


def generate_full_neural_network_configuration(
    cls: type, build_prefix: callable
) -> list[dict]:
    
    encoding_circuit_configs = generate_encoding_circuit_configurations(
        cls, build_prefix
    )
    observable_configs = generate_observable_configurations(cls, build_prefix)

    configurations = []

    # config for all encoding circuits but only one observable
    for encoding_circuit_config in encoding_circuit_configs:
        config = {}
        config.update(encoding_circuit_config)
        config.update(observable_configs[0])
        config.update({build_prefix(cls) + "parameter_seed": 0})
        configurations.append(config)

    # config for all observables but only one encoding circuit
    for observable_config in observable_configs:
        config = {}
        config.update(encoding_circuit_configs[0])
        config.update(observable_config)
        config.update({build_prefix(cls) + "parameter_seed": 0})
        configurations.append(config)

    return configurations


def generate_reservoir_computing_configuration(
    cls: type, build_prefix: callable
) -> list[dict]:
    
    encoding_circuit_configs = generate_encoding_circuit_configurations(
        cls, build_prefix
    )

    ml_model = ["mlp", "linear", "kernel"]
    operators = ["random_paulis", "single_paulis"]
    
    configurations = []

    # config for all encoding circuits but only one observable
    for encoding_circuit_config in encoding_circuit_configs:
        for ml_model_config in ml_model:
            for operator_config in operators:
                config = {}
                config.update(encoding_circuit_config)
                config.update({build_prefix(cls) + "ml_model": ml_model_config})
                config.update({build_prefix(cls) + "operators": operator_config})
                config.update({build_prefix(cls) + "parameter_seed": 0})
                if operator_config == "random_paulis":
                    config.update({build_prefix(cls) + "num_operators": 10})
                    config.update({build_prefix(cls) + "operator_seed": 0})
                configurations.append(config)


    return configurations
##################################################################
# The following tests are classical model tests for classification
##################################################################
@pytest.mark.parametrize(
    "config", generate_svc_configurations(cls=SVC, build_prefix=get_fullname)
)
def test_SVC(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "svc",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=2, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=1), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_ridge_classifier_configurations(cls=RidgeClassifier, build_prefix=get_fullname)
)
def test_RidgeClassifier(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "ridge_classifier",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=2, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_random_forest_classifier_configurations(cls=RandomForestClassifier, build_prefix=get_fullname)
)
def test_RandomForestClassifier(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "random_forest_classifier",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=2, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_perceptron_classifier_configurations(cls=Perceptron, build_prefix=get_fullname)
)
def test_Perceptron(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "perceptron",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=2, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_gaussian_process_classifier_configurations(cls=GaussianProcessClassifier, build_prefix=get_fullname)
)
def test_GaussianProcessClassifier(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "gaussian_process_classifier",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=2, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_decision_tree_classifier_configurations(cls=DecisionTreeClassifier, build_prefix=get_fullname)
)
def test_DecisionTreeClassifier(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "decision_tree_classifier",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
            "autoqml.search_space.classification.classic.decision_tree_classifier.decision_tree_classifier.DecisionTreeClassifier__max_depth": 1,
            "autoqml.search_space.classification.classic.decision_tree_classifier.decision_tree_classifier.DecisionTreeClassifier__min_samples_split": 0.01,
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=2, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=1), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_logistic_regression_configurations(cls=LogisticRegressor, build_prefix=get_fullname)
)
def test_LogisticRegressor(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.ClassificationChoice__choice": "logistic_regression_classifier",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(n_samples=10, n_features=4, n_classes=2)
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(X_train, y_train, timedelta(seconds=1), configuration=config)

    automl = automl.fit(cmd)
    assert automl is not None

##################################################################
# The following tests are classical model tests for regression
##################################################################
@pytest.mark.parametrize(
    "config", generate_svr_configurations(cls=SVR, build_prefix=get_fullname)
)
def test_SVR(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "svr",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=4)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(X_train, y_train, timedelta(seconds=1), configuration=config)

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_random_forest_regression_configurations(cls=RandomForestRegressor, build_prefix=get_fullname)
)
def test_RandomForestRegressor(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "random_forest_regressor",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=4)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(X_train, y_train, timedelta(seconds=1), configuration=config)

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_linear_regression_configurations(cls=LinearRegressor, build_prefix=get_fullname)
)
def test_LinearRegressor(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "linear_regressor",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=4)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(X_train, y_train, timedelta(seconds=1), configuration=config)

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_kernel_ridge_configurations(cls=KernelRidge, build_prefix=get_fullname)
)
def test_KernelRidge(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "kernel_ridge",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=4)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(X_train, y_train, timedelta(seconds=1), configuration=config)

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_decision_tree_regression_configurations(cls=DecisionTreeRegressor, build_prefix=get_fullname)
)
def test_DecisionTreeRegressor(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "decision_tree_regressor",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=4)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(X_train, y_train, timedelta(seconds=1), configuration=config)

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_gaussian_process_regressor_configurations(cls=GaussianProcessRegressor, build_prefix=get_fullname)
)
def test_GaussianProcessRegressor(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "gaussian_process_regressor",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=4)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(X_train, y_train, timedelta(seconds=2), configuration=config)

    automl = automl.fit(cmd)
    assert automl is not None

@pytest.mark.parametrize(
    "config", generate_full_kernel_configurations(cls=QSVR, build_prefix=get_fullname)
)
def test_QSVR(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "qsvr",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )

    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=2)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None


@pytest.mark.parametrize(
    "config", generate_full_kernel_configurations(cls=QGPR, build_prefix=get_fullname)
)
def test_QGPR(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "qgpr",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=2)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None


@pytest.mark.parametrize(
    "config", generate_full_kernel_configurations(cls=QKRR, build_prefix=get_fullname)
)
def test_QKRR(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.regression.RegressionChoice__choice": "qkrr",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=2)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None


@pytest.mark.parametrize(
    "config",
    generate_full_neural_network_configuration(
        cls=QNNRegressor, build_prefix=get_fullname
    ),
)
def test_QNNRegressor(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.regression.RegressionChoice__choice": "qnnr",
            "autoqml.search_space.regression.quantum.qnnr.QNNR__epochs": 2,
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=2)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None


@pytest.mark.parametrize(
    "config",
    generate_reservoir_computing_configuration(
        cls=QRCRegressor, build_prefix=get_fullname
    ),
)
def test_QRCRegressor(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.regression.RegressionChoice__choice": "qrcr",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_regression(n_samples=10, n_features=2)
    automl = AutoQMLTabularRegression()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None


@pytest.mark.parametrize(
    "config",
    generate_full_kernel_configurations(cls=QSVC, build_prefix=get_fullname),
)
def test_QSVC(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "qsvc",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=10, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None


@pytest.mark.parametrize(
    "config",
    generate_full_kernel_configurations(cls=QGPC, build_prefix=get_fullname),
)
def test_QGPC(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "qgpc",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=10, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None


@pytest.mark.parametrize(
    "config",
    generate_full_neural_network_configuration(
        cls=QNNClassifier, build_prefix=get_fullname
    ),
)
def test_QNNClassifier(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.classification.ClassificationChoice__choice": "qnnc",
            "autoqml.search_space.classifictation.quantum.qnnc.QNNC__epochs": 2,
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=10, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None


@pytest.mark.parametrize(
    "config",
    generate_reservoir_computing_configuration(
        cls=QRCClassifier, build_prefix=get_fullname
    ),
)
def test_QRCClassifier(config: dict):
    ray.shutdown()
    config.update(
        {
            "autoqml.search_space.classification.ClassificationChoice__choice": "qrcc",
            "autoqml.search_space.preprocessing.rescaling.RescalingChoice__choice": "min_max_scaling",
            "autoqml.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "no-op",
        }        
    )
    X_train, y_train = sklearn.datasets.make_classification(
        n_samples=10, n_features=4, n_classes=2
    )
    automl = AutoQMLTabularClassification()

    cmd = AutoQMLFitCommand(
        X_train, y_train, timedelta(seconds=2), configuration=config
    )

    automl = automl.fit(cmd)
    assert automl is not None