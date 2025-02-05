from typing import Callable, Union

from optuna import Trial
from sklearn.base import BaseEstimator, ClassifierMixin
from squlearn.optimizers import Adam
from squlearn.qnn import QNNClassifier as squlearn_QNNClassifier
from squlearn.qnn.loss import SquaredLoss

from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin
from autoqml.search_space.util import (
    get_encoding_circuit,
    get_observable,
    sample_encoding_circuit_configuration,
    sample_observable_configuration,
)
from autoqml.util.context import ConfigContext
from autoqml.constants import TrialId


class QNNClassifier(BaseEstimator, ClassifierMixin, TunableMixin):
    def __init__(
        self,
        # encoding circuit related parameters
        encoding_circuit: str = "chebyshev_pqc",
        num_qubits: int = 3,
        num_repetitions: int = 1,
        chebyshev_alpha: float = 1.0,
        num_chebyshev: int = None,
        # observable related parameters
        observable: str = "single_pauli_x",
        observable_qubit: int = 0,
        # others
        learning_rate: float = 0.1,
        epochs: int = 10,
        shuffle: bool = True,
        variance: Union[float, Callable] = None,
        batch_size: int = 10,
        parameter_seed: Union[int, None] = 0,
        trial_id: Union[TrialId, None] = None,
    ):
        # encoding circuit related parameters
        self.encoding_circuit = encoding_circuit
        self.num_qubits = num_qubits
        self.num_repetitions = num_repetitions
        self.chebyshev_alpha = chebyshev_alpha
        self.num_chebyshev = num_chebyshev
        # observable related parameters
        self.observable = observable
        self.observable_qubit = observable_qubit
        # others
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.shuffle = shuffle
        self.variance = variance
        self.batch_size = batch_size
        self.parameter_seed = parameter_seed

        self.estimator = None
        self.trial_id = trial_id

    def fit(self, X: InputData, y: TargetData):

        context: ConfigContext = ConfigContext.instance()
        executor = context.get_config(self.trial_id, "executor")

        # num_features is the dimension of a single data point
        # num_qubits is an attribute of the feature map. However, it can be
        # set and retrieved from the estimator directly as well
        num_features = X.shape[1]

        encoding_circuit = get_encoding_circuit(
            encoding_circuit=self.encoding_circuit,
            num_qubits=self.num_qubits,
            num_features=num_features,
            num_repetitions=self.num_repetitions,
            alpha=self.chebyshev_alpha,
            num_chebyshev=self.num_chebyshev,
        )

        if y.ndim == 1:
            observable = get_observable(
                observable=self.observable,
                num_qubits=self.num_qubits,
                qubit=self.observable_qubit,
            )
        else:
            observable = [
                get_observable(
                    observable=self.observable,
                    num_qubits=self.num_qubits,
                    qubit=self.observable_qubit,
                ) for _ in range(y.shape[1])
            ]

        self.estimator = squlearn_QNNClassifier(
            encoding_circuit=encoding_circuit,
            operator=observable,
            executor=executor,
            loss=SquaredLoss(),
            optimizer=Adam(options={"lr": self.learning_rate}),
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
            variance=self.variance,
            parameter_seed=self.parameter_seed,
        )

        self.estimator.fit(X, y)
        return self

    def predict(self, X: InputData) -> TargetData:
        if not hasattr(self, "estimator"):
            raise NotImplementedError
        return self.estimator.predict(X)

    def sample_configuration(
        self, trial: Trial, defaults: Configuration
    ) -> Configuration:

        config = Configuration({key: None for key in self._get_param_names()})

        config["num_qubits"] = (
            self._get_default_values(trial, 'num_qubits', defaults)
            if self._fullname("num_qubits") in defaults else trial.
            suggest_categorical(self._fullname("num_qubits"), [1, 2, 4, 8])
        )

        config.update(
            sample_encoding_circuit_configuration(
                trial, defaults, self._fullname
            )
        )

        config.update(
            sample_observable_configuration(
                trial, defaults, self._fullname, config["num_qubits"]
            )
        )

        config.update(
            {
                "learning_rate":
                    (
                        self._get_default_values(
                            trial, 'learning_rate', defaults
                        ) if self._fullname("learning_rate") in defaults else
                        trial.suggest_float(
                            self._fullname("learning_rate"), 0.0001, 0.1
                        )
                    ),
                "batch_size":
                    (
                        self.
                        _get_default_values(trial, 'learning_rate', defaults)
                        if self._fullname("batch_size") in defaults else trial.
                        suggest_int(self._fullname("batch_size"), 1, 100)
                    ),
                "epochs":
                    (
                        self._get_default_values(trial, 'epochs', defaults)
                        if self._fullname("epochs") in defaults else
                        trial.suggest_int(self._fullname("epochs"), 5, 100)
                    ),
                "shuffle":
                    (
                        self._get_default_values(trial, 'shuffle', defaults)
                        if self._fullname("shuffle") in defaults else
                        trial.suggest_categorical(
                            self._fullname("shuffle"), [True, False]
                        )
                    ),
                "variance":
                    (
                        self._get_default_values(trial, 'variance', defaults)
                        if self._fullname("variance") in defaults else
                        trial.suggest_categorical(
                            self._fullname("variance"),
                            [None, 0.001, 0.005, 0.0001, 0.0005],
                        )
                    ),
                "parameter_seed":
                    (
                        self._get_default_values(
                            trial, 'parameter_seed', defaults
                        ) if self._fullname("parameter_seed") in defaults else
                        trial.suggest_int(
                            self._fullname("parameter_seed"), 0, 2**32 - 1
                        )
                    ),
            }
        )

        return config
