from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import Configuration
from autoqml_lib.search_space.base import TunableMixin


class LinearRegressor(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, fit_intercept: bool = True, positive: bool = False):
        self.fit_intercept = fit_intercept
        self.positive = positive

    def fit(self, X: InputData, y: TargetData):
        from sklearn.linear_model import LinearRegression

        self.estimator = LinearRegression(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
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
            'fit_intercept':
                (
                    self._get_default_values(trial, 'fit_intercept', defaults)
                    if self._fullname('fit_intercept') in defaults else
                    trial.suggest_categorical(
                        self._fullname('fit_intercept'), [True, False]
                    )
                ),
            'positive':
                (
                    self._get_default_values(trial, 'positive', defaults)
                    if self._fullname('positive')
                    in defaults else trial.suggest_categorical(
                        self._fullname('positive'), [True, False]
                    )
                )
        }
