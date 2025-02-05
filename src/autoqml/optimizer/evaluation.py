import numpy as np

from dataclasses import dataclass
from typing import Callable, Optional, Union

from pennylane.devices import Device as PennylaneDevice
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.providers.backend import Backend
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from sklearn.pipeline import Pipeline
from squlearn import Executor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from autoqml.constants import Budget, InputData, TargetData, TrialId
from autoqml.optimizer.metric import Metric
from autoqml.search_space import Configuration, SearchSpace
from autoqml.search_space.base import TunablePipeline

__all__ = ['evaluate']

from autoqml.util.context import ConfigContext

Score = float


@dataclass
class Trial:
    id: TrialId
    configuration: Configuration
    loss: Optional[Score] = None
    budget: Optional[Budget] = None
    duration: Optional[float] = None


def configuration_to_model(
    configuration: Configuration,
    trial_id: TrialId,
    pipeline_factory: Callable,
) -> TunablePipeline:
    pipeline = pipeline_factory()
    pipeline.trial_id = trial_id
    for _, step in pipeline.steps:
        step.trial_id = trial_id

    # For reasons, ray augments the configuration with additional values
    configuration = {
        k: v
        for k, v in configuration.items() if not k.startswith('autoqml')
    }

    pipeline.set_params(**configuration)
    return pipeline


def construct_executor(**kwargs) -> Executor:
    if isinstance(kwargs["backend"], Executor):
        return kwargs["backend"]
    elif kwargs["backend"] is None:
        return Executor()
    else:
        return Executor(kwargs["backend"])


def fit_configuration(
    X: InputData,
    y: TargetData,
    config: Configuration,
    backend: Union[
        Executor,
        str,
        Backend,
        QiskitRuntimeService,
        Session,
        BaseEstimator,
        BaseSampler,
        PennylaneDevice,
    ],
    pipeline_factory: Callable[..., SearchSpace],
    # The `trial` parameter in the `evaluate` function represents a single trial or experiment that
    # is being evaluated. It contains information such as the trial ID, configuration
    # (hyperparameters), loss value, budget, and duration. The `Trial` dataclass defines the
    # structure of a trial, including these attributes. The `evaluate` function uses the `trial`
    # parameter to access the configuration of the trial being evaluated and to pass it to the
    # `fit_configuration` function for training the pipeline model.
    trial: Optional[Trial] = None,
) -> Union[Pipeline, float]:
    context: ConfigContext = ConfigContext.instance()

    # 1. Create executor
    executor = construct_executor(backend=backend)
    if trial is None:
        trial_id = ''
    else:
        trial_id = trial.id
    context.set_config(trial_id, key='executor', value=executor)

    # 2. Transform Configuration (dict) to TunablePipeline
    # assumption: it does not need to be initialized with steps. it gets the steps from the configuration.
    pipeline = configuration_to_model(config, trial_id, pipeline_factory)

    # 3. Fit pipeline with on train data, later with QML executor
    pipeline.fit(X, y)

    return pipeline


def evaluate(
    trial: Trial,
    X_train: InputData,
    y_train: TargetData,
    X_test: InputData,
    y_test: TargetData,
    backend: Union[
        Executor,
        str,
        Backend,
        QiskitRuntimeService,
        Session,
        BaseEstimator,
        BaseSampler,
        PennylaneDevice,
    ],
    metric: Metric,
    selection: str,
    pipeline_factory: Callable[..., SearchSpace],
) -> Score:

    pipeline = fit_configuration(
        X=X_train,
        y=y_train,
        config=trial.configuration,
        backend=backend,
        pipeline_factory=pipeline_factory,
        trial=trial,
    )
    # See workaround in fit_configuration()
    if isinstance(pipeline, float):
        return pipeline

    # 4. Score solution using metric
    if selection == "split":

        # Compute score on test set
        pred_test = pipeline.predict(X_test)
        score = metric.score(a=pred_test, b=y_test)

    elif selection == "cv":
        # Compute cross-validated score
        scorer = make_scorer(
            metric.score, greater_is_better=not metric.mode_is_minimization
        )
        score = np.mean(
            cross_val_score(pipeline, X_train, y_train, cv=4, scoring=scorer)
        )

        if metric.mode_is_minimization:
            score = -score

        score = np.float64(
            np.where(np.isnan(score), metric.worst_result, score)
        )

    else:
        raise ValueError(f"Selection {selection} not supported")

    # 5. Persist pipeline
    # TODO

    return score
