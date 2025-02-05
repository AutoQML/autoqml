import abc

from sklearn.metrics import accuracy_score, balanced_accuracy_score, root_mean_squared_error, mean_absolute_error

from autoqml.constants import TargetData


class Metric(abc.ABC):
    @abc.abstractmethod
    def score(self, a: TargetData, b: TargetData) -> float:
        raise NotImplementedError()

    worst_result: float
    mode_is_minimization: bool


class Accuracy(Metric):
    def score(self, a: TargetData, b: TargetData) -> float:
        return accuracy_score(a, b)

    worst_result: float = 0.0
    mode_is_minimization: bool = False


class BalancedAccuracy(Metric):
    def score(self, a: TargetData, b: TargetData) -> float:
        return balanced_accuracy_score(a, b)

    worst_result: float = 0.0
    mode_is_minimization: bool = False


class RMSE(Metric):
    def score(self, a: TargetData, b: TargetData) -> float:
        return root_mean_squared_error(a, b)

    worst_result: float = float('inf')
    mode_is_minimization: bool = True


class MAE(Metric):
    def score(self, a: TargetData, b: TargetData) -> float:
        return mean_absolute_error(a, b)

    worst_result: float = float('inf')
    mode_is_minimization: bool = True
