from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class LogisticRegressor(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, penalty: str = 'l2', dual: bool = False,
                C: float = 1.0, fit_intercept: bool = True,
                max_iter: int = 100):
        self.penalty = penalty
        self.dual = dual
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X: InputData, y: TargetData):
        from sklearn.linear_model import LogisticRegression

        self.estimator = LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter
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
        return {
            'penalty':
                (
                    self._get_default_values(trial, 'penalty', defaults)
                    if self._fullname('penalty') in defaults else
                    trial.suggest_categorical(
                        self._fullname('penalty'), [None, 'l2', 'l1', 'elasticnet']
                    )
                ),
            'dual':
                (
                    self._get_default_values(trial, 'dual', defaults)
                    if self._fullname('dual')
                    in defaults else trial.suggest_categorical(
                        self._fullname('dual'), [True, False]
                    )
                ),
            'C':
                (
                    self._get_default_values(trial, 'C', defaults)
                    if self._fullname('C') in defaults else
                    trial.suggest_float(
                        self._fullname('C'), 0.01, 10000, log=True
                    )
                ),
            'fit_intercept':
                (
                    self._get_default_values(trial, 'fit_intercept', defaults)
                    if self._fullname('fit_intercept')
                    in defaults else trial.suggest_categorical(
                        self._fullname('fit_intercept'), [True, False]
                    )
                ),
            'max_iter':
                (
                    self._get_default_values(trial, 'max_iter', defaults)
                    if self._fullname('max_iter')
                    in defaults else trial.suggest_int(
                        self._fullname('max_iter'), 10, 10000, log=True
                    )
                )
        }
