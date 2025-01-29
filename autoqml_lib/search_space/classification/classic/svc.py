from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import Configuration
from autoqml_lib.search_space.base import TunableMixin


class SVC(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, C: float = 0.1, kernel: str = 'rbf'):
        self.C = C
        self.kernel = kernel

    def fit(self, X: InputData, y: TargetData):
        from sklearn.svm import SVC

        self.estimator = SVC(
            C=self.C,
            kernel=self.kernel,
        )

        self.estimator.fit(X, y)
        return self

    def predict(self, X: InputData) -> TargetData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError
        return self.estimator.predict(X)

    def sample_configuration(
            self,
            trial: Trial,
            defaults: Configuration,
            dataset_statistics: DataStatistics
    ) -> Configuration:
        return {
            'kernel': (
                self._get_default_values(trial, 'kernel', defaults)
                if self._fullname('kernel') in defaults
                else trial.suggest_categorical(self._fullname('kernel'), ['rbf', 'poly', 'sigmoid'])
            ),
            'C': (
                self._get_default_values(trial, 'C', defaults)
                if self._fullname('C') in defaults
                else trial.suggest_float(self._fullname('C'), 0.03125, 32768, log=True)
            )
        }
