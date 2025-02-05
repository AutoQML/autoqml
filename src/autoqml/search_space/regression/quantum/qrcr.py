from typing import Union

from optuna import Trial
from sklearn.base import BaseEstimator, RegressorMixin

from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin
from autoqml.search_space.util import (
    get_encoding_circuit,
    sample_encoding_circuit_configuration,
)
from autoqml.util.context import ConfigContext
from autoqml.constants import TrialId


class QRCRegressor(BaseEstimator, RegressorMixin, TunableMixin):
    def __init__(
        self,
        # encoding circuit related parameters
        encoding_circuit: str = "chebyshev_pqc",
        num_qubits: int = 3,
        num_repetitions: int = 1,
        chebyshev_alpha: float = 1.0,
        num_chebyshev: int = None,
        # qrc related parameters
        ml_model: str = "linear",
        operators: str = "random_paulis",
        num_operators: int = 100,
        operator_seed: int = 0,
        parameter_seed: int = 0,
        trial_id: Union[TrialId, None] = None,
    ):
        # encoding circuit related parameters
        self.encoding_circuit = encoding_circuit
        self.num_qubits = num_qubits
        self.num_repetitions = num_repetitions
        self.chebyshev_alpha = chebyshev_alpha
        self.num_chebyshev = num_chebyshev
        # qrc related parameters
        self.ml_model = ml_model
        self.operators = operators
        self.num_operators = num_operators
        self.operator_seed = operator_seed
        self.parameter_seed = parameter_seed
        self.trial_id = trial_id

    def fit(self, X: InputData, y: TargetData):
        from squlearn.qrc import QRCRegressor

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

        self.estimator = QRCRegressor(
            encoding_circuit=encoding_circuit,
            executor=executor,
            ml_model=self.ml_model,
            operators=self.operators,
            num_operators=self.num_operators,
            operator_seed=self.operator_seed,
            parameter_seed=self.parameter_seed
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
            defaults[self._fullname("num_qubits")]
            if self._fullname("num_qubits") in defaults else trial.
            suggest_categorical(self._fullname("num_qubits"), [1, 2, 4, 8])
        )

        config.update(
            sample_encoding_circuit_configuration(
                trial, defaults, self._fullname
            )
        )

        config.update(
            {
                "ml_model":
                    (
                        self._get_default_values(trial, 'ml_model', defaults)
                        if self._fullname("ml_model") in defaults else
                        trial.suggest_categorical(
                            self._fullname("ml_model"),
                            ["mlp", "linear", "kernel"]
                        )
                    ),
                "operators":
                    (
                        self._get_default_values(trial, 'operators', defaults)
                        if self._fullname("operators") in defaults else
                        trial.suggest_categorical(
                            self._fullname("operators"),
                            ["random_paulis", "single_paulis"]
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

        if config["operators"] == "random_paulis":
            config.update(
                {
                    "num_operators":
                        (
                            self._get_default_values(
                                trial, 'num_operators', defaults
                            ) if self._fullname("num_operators") in defaults
                            else trial.suggest_int(
                                self._fullname("num_operators"),
                                1,
                                10000,
                                log=True
                            )
                        ),
                    "operator_seed":
                        (
                            self._get_default_values(
                                trial, 'operator_seed', defaults
                            ) if self._fullname("operator_seed") in defaults
                            else trial.suggest_int(
                                self._fullname("operator_seed"), 0, 2**32 - 1
                            )
                        ),
                }
            )

        return config
