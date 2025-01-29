from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import Configuration
from autoqml_lib.search_space.base import TunableMixin


class TSNE(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, n_components: float = 0.1):
        self.n_components = n_components

    def fit(self, X: InputData, y: TargetData):
        from sklearn.manifold import TSNE

        perplexity = max(1,min(30, int(X.shape[0]*0.2)-1))  # 0.2 train test split

        tsne_components = max(1, int(self.n_components))
        self.estimator = TSNE(n_components=tsne_components, perplexity=perplexity)
        self.estimator.fit(X, y)
        return self

    def transform(self, X: InputData) -> InputData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError()

        X = self.estimator.fit_transform(X)
        return X

    def sample_configuration(
            self,
            trial: Trial,
            defaults: Configuration,
            dataset_statistics: DataStatistics
    ) -> Configuration:
        return {
            'n_components': (
                self._get_default_values(trial, 'n_components', defaults)
                if self._fullname('n_components') in defaults
                else trial.suggest_float(self._fullname('n_components'), 0.0, 1.0)
            )
        }
