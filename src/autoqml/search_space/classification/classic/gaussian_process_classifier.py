from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class GaussianProcessClassifier(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(
        self,
        kernel: str = 'RBF',
        warm_start: bool = False,
        multi_class: str = 'one_vs_rest'
    ):
        self.kernel = kernel
        self.warm_start = warm_start
        self.multi_class = multi_class

    def fit(self, X: InputData, y: TargetData):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern

        kernel = None
        if self.kernel == 'RBF':
            kernel = RBF()
        elif self.kernel == 'DotProduct':
            kernel = DotProduct()
        elif self.kernel == 'Matern':
            kernel = Matern()

        self.estimator = GaussianProcessClassifier(
            kernel=kernel,
            warm_start=self.warm_start,
            multi_class=self.multi_class
        )

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
            'kernel':
                (
                    self._get_default_values(trial, 'kernel', defaults)
                    if self._fullname('kernel') in defaults else
                    trial.suggest_categorical(
                        self._fullname('kernel'),
                        ['RBF', 'DotProduct', 'Matern']
                    )
                ),
            'warm_start':
                (
                    self._get_default_values(trial, 'warm_start', defaults)
                    if self._fullname('warm_start') in defaults else
                    trial.suggest_categorical(
                        self._fullname('warm_start'), [True, False]
                    )
                ),
            'multi_class':
                (
                    self._get_default_values(trial, 'multi_class', defaults)
                    if self._fullname('multi_class') in defaults else
                    trial.suggest_categorical(
                        self._fullname('multi_class'),
                        ['one_vs_rest', 'one_vs_one']
                    )
                )
        }
