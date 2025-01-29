from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import Configuration
from autoqml_lib.search_space.base import TunableMixin


class KernelRidge(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, alpha: float = 0.1, kernel: str = 'rbf', gamma: float = 1, degree: int = 3, coef0: float = 1):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
    
    def fit(self, X: InputData, y: TargetData):
        from sklearn.kernel_ridge import KernelRidge

        self.estimator = KernelRidge(
            alpha=self.alpha,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0
        )

        self.estimator.fit(X, y)
        return self

    def predict(self, X: InputData) -> TargetData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError
        return self.estimator.predict(X)

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        base_config = {
            'kernel':
                (
                    self._get_default_values(trial, 'kernel', defaults)
                    if self._fullname('kernel') in defaults else
                    trial.suggest_categorical(
                        self._fullname('kernel'), ['rbf', 'poly', 'sigmoid']
                    )
                ),
            'alpha':
                (
                    self._get_default_values(trial, 'alpha', defaults)
                    if self._fullname('alpha')
                    in defaults else trial.suggest_float(
                        self._fullname('alpha'), 0.03125, 32768, log=True
                    )
                )       
        }
        
        if base_config['kernel'] == "rbf":
            base_config['gamma'] = (
                    self._get_default_values(trial, 'gamma', defaults)
                    if self._fullname('gamma')
                    in defaults else trial.suggest_float(
                        self._fullname('gamma'), 0.0001, 10000, log=True
                    )
                )
        if base_config['kernel'] == "poly":
            base_config['degree'] = (
                    self._get_default_values(trial, 'degree', defaults)
                    if self._fullname('degree')
                    in defaults else trial.suggest_int(
                        self._fullname('degree'), 1, 10
                    )
                )
        if base_config['kernel'] == "poly" or base_config['kernel'] == "sigmoid":
            base_config['coef0'] = (
                    self._get_default_values(trial, 'coef0', defaults)
                    if self._fullname('coef0')
                    in defaults else trial.suggest_float(
                        self._fullname('coef0'), 0.03125, 32768, log=True
                    )
                )

        return base_config
        
