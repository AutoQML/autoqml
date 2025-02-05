from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class RidgeClassifier(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: InputData, y: TargetData):
        from sklearn.linear_model import RidgeClassifier

        self.estimator = RidgeClassifier(
            alpha=self.alpha, fit_intercept=self.fit_intercept
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
            'alpha':
                (
                    self._get_default_values(trial, 'alpha', defaults)
                    if self._fullname('alpha') in defaults else
                    trial.suggest_float(
                        self._fullname('alpha'), 0.03125, 32768, log=True
                    )
                ),
            'fit_intercept':
                (
                    self._get_default_values(trial, 'fit_intercept', defaults)
                    if self._fullname('fit_intercept') in defaults else
                    trial.suggest_categorical(
                        self._fullname('fit_intercept'), [True, False]
                    )
                )
        }
