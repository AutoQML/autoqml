from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml.constants import InputData, TargetData
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class GaussianProcessRegressor(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, kernel: str ='RBF',  alpha: float = 0.1, normalize_y: bool = False):
        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        

    def fit(self, X: InputData, y: TargetData):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern
        
        kernel = None
        if self.kernel == 'RBF':
            kernel = RBF()
        elif self.kernel == 'DotProduct':
            kernel = DotProduct()
        elif self.kernel == 'Matern':
            kernel = Matern()        

        self.estimator = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y
        )

        self.estimator.fit(X, y)
        return self

    def predict(self, X: InputData) -> TargetData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError
        return self.estimator.predict(X)

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:        
        from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern
        return {
            'kernel':
                (
                    self._get_default_values(trial, 'kernel', defaults)
                    if self._fullname('kernel') in defaults else
                    trial.suggest_categorical(
                        self._fullname('kernel'), ['RBF', 'DotProduct', 'Matern']
                    )
                ),
            'alpha':
                (
                    self._get_default_values(trial, 'alpha', defaults)
                    if self._fullname('alpha')
                    in defaults else trial.suggest_float(
                        self._fullname('alpha'), 0.03125, 32768, log=True
                    )
                ),
            'normalize_y':
                (
                    self._get_default_values(trial, 'normalize_y', defaults)
                    if self._fullname('normalize_y')
                    in defaults else trial.suggest_categorical(
                        self._fullname('normalize_y'), [True, False]
                    )
                ),
        }
