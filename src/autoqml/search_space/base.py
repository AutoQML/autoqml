import abc
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional

import numpy as np
from optuna import Trial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from autoqml.constants import TrialId, InputData, TargetData
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import Configuration


def sub_configuration(params: Configuration) -> tuple[str, Configuration]:
    new_params = {}
    choice: str = params['choice']

    for param, value in params.items():
        if param == 'choice' or (not param.startswith(f'{choice}__')):
            continue
        param = param.replace(f'{choice}__', '', 1)
        new_params[param] = value

    return choice, new_params


class TunableMixin:
    trial_id: TrialId

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        pass

    def _fullname(self, suffix: str) -> str:
        cls = type(self)
        module = cls.__module__
        name = cls.__qualname__
        if module is not None and module != '__builtin__':
            name = f'{module}.{name}'
        return f'{name}__{suffix}'

    def _get_default_values(self, trial:Trial, suffix:str, defaults: Configuration):
        """ Helper function to handle default sequences """
        return (trial.suggest_categorical(self._fullname(suffix),defaults[self._fullname(suffix)])
                if isinstance(defaults[self._fullname(suffix)],Sequence) and not isinstance(defaults[self._fullname(suffix)], (str, bytes))
                else defaults[self._fullname(suffix)]
                )


class EstimatorChoice(BaseEstimator, TunableMixin, abc.ABC):
    def __init__(
        self,
        estimator: BaseEstimator = None,
        random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, *args, **kwargs):
        self.estimator.fit(*args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        return self.estimator.transform(*args, **kwargs)

    @classmethod
    @abc.abstractmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        raise NotImplementedError()

    def get_available_components(
        self,
        default: str = None,
        include: list[str] = None,
        exclude: list[str] = None
    ) -> tuple[dict[str, type[TunableMixin]], Optional[str]]:
        if include is not None and exclude is not None:
            raise ValueError(
                'The argument include and exclude cannot be used together.'
            )

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError(
                        f'Trying to include unknown component: {incl}'
                    )

        components_dict = OrderedDict()
        for name, entry in available_comp.items():
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue
            components_dict[name] = available_comp[name]

        if len(components_dict) == 0:
            raise ValueError('No estimators found')

        if default is None or default not in components_dict.keys():
            for default_ in components_dict.keys():
                if include is not None and default_ not in include:
                    continue
                if exclude is not None and default_ in exclude:
                    continue
                default = default_
                break

        return components_dict, default

    def sample_configuration(
        self,
        trial: Trial,
        defaults: Configuration,
        dataset_statistics: DataStatistics,
        default: str = None,
        include: list[str] = None,
        exclude: list[str] = None
    ) -> Configuration:
        available_components, _ = self.get_available_components(
            default, include, exclude
        )

        choice = (
            self._get_default_values(trial, 'choice', defaults)
            if self._fullname('choice')
            in defaults else trial.suggest_categorical(
                self._fullname('choice'), list(available_components.keys())
            )
        )
        config = {
            f'{choice}__{k}': v
            for k, v in available_components[choice]().
            sample_configuration(trial, defaults, dataset_statistics).items()
        }
        config['choice'] = choice
        return config

    def set_params(self, **configuration):
        choice, new_params = sub_configuration(configuration)
        # noinspection PyArgumentList
        self.estimator = self.get_components()[choice](**new_params)
        self.estimator.trial_id = self.trial_id
        return self


class TunablePipeline(Pipeline, TunableMixin):
    trial_id: TrialId

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        config = {}
        for name, estimator in self.steps:
            comp_config = {
                f'{name}__{k}': v
                for k, v in estimator.sample_configuration(
                    trial, defaults, dataset_statistics
                ).items()
            }
            config.update(comp_config)
        return config


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: InputData, y: TargetData = None):
        return self

    def transform(self, X: InputData, y: TargetData = None):
        return X
