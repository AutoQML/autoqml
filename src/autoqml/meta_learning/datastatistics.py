import abc
from dataclasses import dataclass

from autoqml.constants import InputData


@dataclass
class DataStatistics(abc.ABC):
    n_samples: int
    n_features: int


class FeatureExtractor(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def calculate(self, X: InputData) -> DataStatistics:
        pass


@dataclass
class TabularStatistics(DataStatistics):
    pass


class TabularFeatureExtractor(FeatureExtractor):
    def calculate(self, X: InputData) -> TabularStatistics:
        # 1. Extract meta-features from InputData
        pass
