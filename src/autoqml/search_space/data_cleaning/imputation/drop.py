import numpy as np
import pandas as pd
from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import SearchSpace, Configuration
from autoqml.search_space.base import TunableMixin


class DropImputation(BaseEstimator, TransformerMixin, TunableMixin):
    def fit(self, X: InputData, y: TargetData):
        pass

    def transform(self, X: InputData) -> InputData:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X = X.dropna()
        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        return {}
