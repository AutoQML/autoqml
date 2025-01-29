from typing import Union
from optuna import Trial
from sklearn.base import TransformerMixin, BaseEstimator

from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import Configuration
from autoqml_lib.search_space.base import TunableMixin


class DecisionTreeClassifier(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(self, criterion: str = 'gini',
                 splitter: str = 'best',
                 max_depth: int = None, 
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: InputData, y: TargetData):
        from sklearn.tree import DecisionTreeClassifier

        self.estimator = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
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
        return {
            'criterion':
                (
                    self._get_default_values(trial, 'criterion', defaults)
                    if self._fullname('criterion') in defaults else
                    trial.suggest_categorical(
                        self._fullname('criterion'), ['gini', 'entropy', 'log_loss']
                    )
                ),
            'splitter':
                (
                    self._get_default_values(trial, 'splitter', defaults)
                    if self._fullname('splitter') in defaults else
                    trial.suggest_categorical(
                        self._fullname('splitter'), ['best', 'random']
                    )
                ),
            'max_depth':
                (
                    self._get_default_values(trial, 'max_depth', defaults)
                    if self._fullname('max_depth') in defaults else 
                    trial.suggest_int(
                        self._fullname('max_depth'), 1, 1000, log=False
                    )
                ),
            'min_samples_split':
                (
                    self._get_default_values(trial, 'min_samples_split', defaults)
                    if self._fullname('min_samples_split')
                    in defaults else trial.suggest_float(
                        self._fullname('min_samples_split'), 0.01, 1.0, log=True
                    )
                ),
            'min_samples_leaf':
                (
                    self._get_default_values(trial, 'min_samples_leaf', defaults)
                    if self._fullname('min_samples_leaf')
                    in defaults else trial.suggest_float(
                        self._fullname('min_samples_leaf'), 0.01, 1.0, log=True
                    )
                )
        }
