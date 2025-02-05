from typing import Union

from optuna import Trial
from sklearn.base import BaseEstimator, ClassifierMixin
from squlearn.kernel.matrix.projected_quantum_kernel import OuterKernelBase
from squlearn.observables.observable_base import ObservableBase

from autoqml.constants import InputData, TargetData
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin
from autoqml.search_space.util import (
    get_encoding_circuit,
    get_quantum_kernel,
    sample_encoding_circuit_configuration,
    sample_quantum_kernel_configuration,
)
from autoqml.util.context import ConfigContext
from autoqml.constants import TrialId


class QSVC(BaseEstimator, ClassifierMixin, TunableMixin):
    def __init__(
        self,
        # encoding circuit related parameters
        encoding_circuit: str = "chebyshev_pqc",
        num_qubits: int = 3,
        num_repetitions: int = 1,
        chebyshev_alpha: float = 1.0,
        num_chebyshev: int = None,
        # kernel related parameters
        quantum_kernel: str = "projected_quantum_kernel",
        measurement: Union[str, ObservableBase, list] = "X",
        outer_kernel: Union[str, OuterKernelBase] = "gaussian",
        parameter_seed: Union[int, None] = 0,
        # others
        C: float = 100,
        trial_id: Union[TrialId, None] = None,
    ):
        # encoding circuit related parameters
        self.encoding_circuit = encoding_circuit
        self.num_qubits = num_qubits
        self.num_repetitions = num_repetitions
        self.chebyshev_alpha = chebyshev_alpha
        self.num_chebyshev = num_chebyshev
        # kernel related parameters
        self.quantum_kernel = quantum_kernel
        self.measurement = measurement
        self.outer_kernel = outer_kernel
        self.parameter_seed = parameter_seed
        # others
        self.C = C
        self.trial_id = trial_id

    def fit(self, X: InputData, y: TargetData):
        from squlearn.kernel import QSVC

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

        quantum_kernel = get_quantum_kernel(
            quantum_kernel=self.quantum_kernel,
            encoding_circuit=encoding_circuit,
            executor=executor,
            measurement=self.measurement,
            outer_kernel=self.outer_kernel,
            parameter_seed=self.parameter_seed,
        )

        self.estimator = QSVC(quantum_kernel=quantum_kernel, C=self.C)

        self.estimator.fit(X, y)
        return self

    def predict(self, X: InputData) -> TargetData:
        if not hasattr(self, "estimator"):
            raise NotImplementedError
        return self.estimator.predict(X)

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
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
            sample_quantum_kernel_configuration(
                trial, defaults, self._fullname
            )
        )

        config.update(
            {
                "C":
                    (
                        self._get_default_values(trial, 'C', defaults)
                        if self._fullname("C") in defaults else
                        trial.suggest_float(
                            self._fullname("C"), 0.03125, 32768, log=True
                        )
                        # self._fullname('C'), 90, 110, step=10)  # Fraunhofer C == 100
                    ),
            }
        )

        return config
