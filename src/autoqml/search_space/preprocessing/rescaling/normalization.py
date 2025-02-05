from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class Normalization(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, norm: str = 'l2'):
        self.norm = norm

    def fit(self, X: InputData, y: TargetData):
        from sklearn.preprocessing import Normalizer

        self.estimator = Normalizer(norm=self.norm)
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
            'norm':
                (
                    self._get_default_values(trial, 'norm', defaults)
                    if self._fullname('norm') in defaults else
                    trial.suggest_categorical(
                        self._fullname('norm'), ['l1', 'l2', 'max']
                    )
                )
        }
