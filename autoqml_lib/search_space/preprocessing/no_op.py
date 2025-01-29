from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from optuna import Trial

from autoqml_lib.search_space import Configuration
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.search_space.base import TunableMixin


class NoOp(BaseEstimator, TransformerMixin, TunableMixin):
    def fit(self, X: InputData, y: TargetData):
        return self

    def transform(
        self, X: InputData, y: Optional[TargetData] = None
    ) -> InputData:
        return X

    def sample_configuration(
        self,
        trial: Trial,
        defaults: Configuration,
        dataset_statistics: DataStatistics,
    ) -> Configuration:

        return {}
