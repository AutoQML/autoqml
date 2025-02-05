import logging
from typing import Optional

from optuna import Trial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

from autoqml.constants import InputData, TargetData
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin


class ResamplingException(Exception):
    pass


class MissingTargetsForStratification(ResamplingException):
    pass


class Resampling(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(
        self,
        stratify: bool = False,
        n_samples: Optional[int] = None,
    ):
        """
        n_samples int, default=None
        Number of samples to generate.
        If left to None this is automatically set to the
        first dimension of the arrays.
        If replace is False it should not be larger than
        the length of arrays.
        """
        self.n_samples = n_samples
        self.stratify = stratify

    def fit(self, X: InputData, y: TargetData):
        pass

    def transform(
        self, X: InputData, y: Optional[TargetData] = None
    ) -> InputData:
        if self.stratify:
            if not y:
                # raise MissingTargetsForStratification(
                #     'Resampling.stratify is True, but the y target for'
                #     'stratification is missing.'
                # )
                logging.error(
                    'Resampling.stratify is True, but the y target for'
                    'stratification is missing.'
                    'Returning X unchanged.'
                )
                return X
            X = resample(
                X,
                n_samples=self.n_samples,
                stratify=y,
            )
        else:
            X = resample(X, n_samples=self.n_samples, stratify=None)
        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        return {
            'stratify':
                (
                    self._get_default_values(trial, 'stratify', defaults)
                    if self._fullname('stratify') in defaults else
                    trial.suggest_categorical(
                        self._fullname('stratify'), [True, False]
                    )
                )
        }
