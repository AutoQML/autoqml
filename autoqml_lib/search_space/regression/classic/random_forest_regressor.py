from typing import Union
from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import Configuration
from autoqml_lib.search_space.base import TunableMixin


class RandomForestRegressor(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, n_estimators: int = 100, criterion: str = 'squared_error',
                max_depth: int = None, min_samples_split: Union[int, float] = 2,
                min_samples_leaf: Union[int, float] = 1):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: InputData, y: TargetData):
        from sklearn.ensemble import RandomForestRegressor

        self.estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
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
            'criterion':
                (
                    self._get_default_values(trial, 'criterion', defaults)
                    if self._fullname('criterion') in defaults else
                    trial.suggest_categorical(
                        self._fullname('criterion'), ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
                    )
                ),
            'n_estimators':
                (
                    self._get_default_values(trial, 'n_estimators', defaults)
                    if self._fullname('n_estimators')
                    in defaults else trial.suggest_int(
                        self._fullname('n_estimators'), 10, 10000, log=True
                    )
                ),
            'max_depth':
                (
                    self._get_default_values(trial, 'max_depth', defaults)
                    if self._fullname('max_depth')
                    in defaults else trial.suggest_int(
                        self._fullname('max_depth'), 1, 1000, log=False
                    )
                ),
            'min_samples_split':
                (
                    self._get_default_values(trial, 'min_samples_split', defaults)
                    if self._fullname('min_samples_split')
                    in defaults else trial.suggest_float(
                        self._fullname('min_samples_split'), 0.01, 1.0, log=True
                    )
                ),
            'min_samples_leaf':
                (
                    self._get_default_values(trial, 'min_samples_leaf', defaults)
                    if self._fullname('min_samples_leaf')
                    in defaults else trial.suggest_float(
                        self._fullname('min_samples_leaf'), 0.01, 1.0, log=True
                    )
                )
        }
