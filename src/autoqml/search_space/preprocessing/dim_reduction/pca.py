from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class PCA(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, n_components: float = 0.1):
        self.n_components = n_components

    def fit(self, X: InputData, y: TargetData):
        from sklearn.decomposition import PCA

        self.estimator = PCA(n_components=self.n_components)
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
        return {
            'n_components':
                (
                    self._get_default_values(trial, 'n_components', defaults)
                    if self._fullname('n_components') in defaults else trial.
                    suggest_float(self._fullname('n_components'), 0.0, 1.0)
                )
        }
