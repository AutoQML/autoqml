import abc
from dataclasses import dataclass


@dataclass
class DataStatistics(abc.ABC):
    n_samples: int
    n_features: int


@dataclass
class TabularStatistics(DataStatistics):
    pass
