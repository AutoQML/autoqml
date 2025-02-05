from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class Perceptron(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, penalty: str = None, alpha: float = 0.001):
        self.alpha = alpha
        self.penalty = penalty

    def fit(self, X: InputData, y: TargetData):
        from sklearn.linear_model import Perceptron

        self.estimator = Perceptron(alpha=self.alpha, penalty=self.penalty)

        self.estimator.fit(X, y)
        return self

    def predict(self, X: InputData) -> TargetData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError
        return self.estimator.predict(X)

    def sample_configuration(
        self, trial: Trial, defaults: Configuration
    ) -> Configuration:
        return {
            'penalty':
                (
                    self._get_default_values(trial, 'penalty', defaults)
                    if self._fullname('penalty') in defaults else
                    trial.suggest_categorical(
                        self._fullname('penalty'), [None, 'l2', 'l1']
                    )
                ),
            'alpha':
                (
                    self._get_default_values(trial, 'alpha', defaults)
                    if self._fullname('alpha') in defaults else
                    trial.suggest_float(
                        self._fullname('alpha'), 0.03125, 32768, log=True
                    )
                )
        }
