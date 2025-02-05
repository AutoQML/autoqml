import pandas as pd
from optuna import Trial
from ray import tune
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import SearchSpace, Configuration
from autoqml.search_space.base import TunableMixin


class ConstantImputation(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, constant: float = 0.0):
        self.constant = constant

    def fit(self, X: InputData, y: TargetData):
        pass

    def transform(self, X: InputData) -> InputData:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.fillna(self.constant)
        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        return {
            'constant':
                (
                    self._get_default_values(trial, 'constant', defaults)
                    if self._fullname('constant') in defaults else trial.
                    suggest_categorical(self._fullname('constant'), [0, 1, 2])
                )
        }
