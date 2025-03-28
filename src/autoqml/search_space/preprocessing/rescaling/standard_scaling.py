from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class StandardScaling(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self):
        pass

    def fit(self, X: InputData, y: TargetData):
        from sklearn.preprocessing import StandardScaler

        self.estimator = StandardScaler()
        self.estimator.fit(X, y)
        return self

    def transform(self, X: InputData) -> InputData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError()

        X = self.estimator.transform(X)
        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration
    ) -> Configuration:
        return {}
