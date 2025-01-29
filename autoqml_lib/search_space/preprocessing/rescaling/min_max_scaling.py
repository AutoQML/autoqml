from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import Configuration
from autoqml_lib.search_space.base import TunableMixin


class MinMaxScaling(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, feature_range: float = 1):
        self.feature_range = feature_range

    def fit(self, X: InputData, y: TargetData):
        from sklearn.preprocessing import MinMaxScaler

        self.estimator = MinMaxScaler(feature_range=(0, self.feature_range))
        self.estimator.fit(X, y)
        return self

    def transform(self, X: InputData) -> InputData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError()

        X = self.estimator.transform(X)
        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        return {
            'feature_range':
                (
                    self._get_default_values(trial, 'feature_range', defaults)
                    if self._fullname('feature_range') in defaults else trial.
                    suggest_float(self._fullname('feature_range'), 0.0, 1.0)
                )
        }


class MinMaxScalingForQuantumKernel(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, range_factor: float = 1):
        self.range_factor = range_factor

    def fit(self, X: InputData, y: TargetData):
        from math import pi
        from sklearn.preprocessing import MinMaxScaler

        self.estimator = MinMaxScaler(feature_range=(-pi*self.range_factor, pi*self.range_factor))
        self.estimator.fit(X, y)
        return self

    def transform(self, X: InputData) -> InputData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError()

        X = self.estimator.transform(X)
        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        return {
            'range_factor':
                (
                    self._get_default_values(trial, 'range_factor', defaults)
                    if self._fullname('range_factor') in defaults else trial.
                    suggest_float(self._fullname('range_factor'), 0.0, 1.0)
                )
        }
