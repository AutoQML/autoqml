import numpy as np
import pandas as pd
from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.search_space import SearchSpace, Configuration
from autoqml.search_space.base import TunableMixin


class MeanImputation(BaseEstimator, TransformerMixin, TunableMixin):
    def fit(self, X: InputData, y: TargetData):
        pass

    def transform(self, X: InputData) -> InputData:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        for i in X.columns[X.isnull().any(axis=0)]:
            X[i].fillna(X[i].mean(), inplace=True)

        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration
    ) -> Configuration:
        return {}
