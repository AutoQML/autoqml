from dataclasses import dataclass
from datetime import timedelta
from typing import Union

from optuna.samplers import BaseSampler as OptunaBaseSampler

from pennylane.devices import Device as PennylaneDevice
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.providers.backend import Backend
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from squlearn import Executor

from autoqml.backend import Backend
from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration


@dataclass
class Message():
    ...


@dataclass
class Event(Message):
    ...


@dataclass
class Command(Message):
    ...


@dataclass
class AutoQMLFitCommand(Command):
    X: InputData
    y: TargetData
    time_budget_for_this_task: timedelta
    configuration: Union[str, Configuration, None] = None
    use_multifidelity: bool = False
    backend: Union[
        Executor,
        str,
        Backend,
        QiskitRuntimeService,
        Session,
        BaseEstimator,
        BaseSampler,
        PennylaneDevice,
        None
    ] = None
    seed: Union[int, None] = 0
    sampler: Union[OptunaBaseSampler, None] = None
    num_startup_trials: int = 100
    log_file: Union[str, None] = "autoqml"
    time_budget_for_trials: Union[timedelta, None] = timedelta(minutes=5)
    selection: str = "split" # or cv or time_ordered
