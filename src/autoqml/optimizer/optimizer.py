import abc
import logging
import time
import warnings
import uuid
from datetime import datetime, timedelta
import queue
import multiprocessing
from typing import Any, Callable, Dict, Optional, Union

import optuna
import ray
import queue
import copy
import os
import sys
from functools import partial
from optuna import Trial as OptunaTrial
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.samplers import BaseSampler as OptunaBaseSampler
from pennylane.devices import Device as PennylaneDevice
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.providers.backend import Backend
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from ray import air, tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.optuna.optuna_search import (
    DEFINE_BY_RUN_WARN_THRESHOLD_S, _OptunaTrialSuggestCaptor
)
from sklearn.model_selection import train_test_split
from squlearn import Executor

from autoqml.constants import InputData, TargetData
from autoqml.messages import AutoQMLFitCommand
from autoqml.optimizer import evaluation, metric
from autoqml.search_space import Configuration, SearchSpace


def is_single_configuration(space_fn: Callable, func_kwargs: dict) -> bool:
    """
    Function for checking if there are multiple configurations available in the search space
    Used to filter the edge case, that only a single configuration is possible
    """
    class StaticTrial(optuna.trial.Trial):
        def __init__(self):
            self._params = {}

        def suggest_float(self, name, low, high, *args, **kwargs):
            if low != high:
                raise ValueError(f"{name} has a float range: {low} to {high}")
            self._params[name] = low
            return low

        def suggest_int(self, name, low, high, *args, **kwargs):
            if low != high:
                raise ValueError(f"{name} has an int range: {low} to {high}")
            self._params[name] = low
            return low

        def suggest_categorical(self, name, choices):
            if len(choices) > 1:
                raise ValueError(f"{name} has multiple categorical choices: {choices}")
            self._params[name] = choices[0]
            return choices[0]

    try:
        trial = StaticTrial()
        space_fn(trial, **func_kwargs)
        return True  # No errors → only one config
    except ValueError:
        return False


class OutputControl:
    """ 
    Class for switch off the output of the AutoQML library
    """
    def __init__(self):
        self.original_stdout_fd = None
        self.original_stderr_fd = None

        self.optuna_logging = None
        self.lightning_fabric = None
        self.lightning_pytorch = None

    def output_off(self):
        """ Switch off the outputs """

        self.optuna_logging = logging.getLogger("optuna").getEffectiveLevel()
        logging.getLogger("optuna").setLevel(logging.WARNING)

        self.lightning_fabric = logging.getLogger("lightning.fabric"
                                                 ).getEffectiveLevel()
        logging.getLogger("lightning.fabric").setLevel(logging.WARNING)

        self.lightning_pytorch = logging.getLogger("lightning.pytorch"
                                                  ).getEffectiveLevel()
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

        self.original_stdout_fd = copy.copy(
            os.dup(1)
        )  # Save file descriptor 1 (`stdout`)
        self.original_stderr_fd = copy.copy(
            os.dup(2)
        )  # Save file descriptor 2 (`stderr`)

        devnull = open(os.devnull, 'w')
        sys.stderr = devnull
        sys.stdout = devnull

        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(
            devnull, 1
        )  # Replace file descriptor 1 (stdout) with `/dev/null`
        os.dup2(
            devnull, 2
        )  # Replace file descriptor 2 (stderr) with `/dev/null`
        os.close(devnull)
        return self.original_stdout_fd

    def output_on(self):
        """ Restore the outputs """
        os.dup2(self.original_stdout_fd, 1)
        os.close(self.original_stdout_fd)
        os.dup2(self.original_stderr_fd, 2)
        os.close(self.original_stderr_fd)
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__

        logging.getLogger("optuna").setLevel(self.optuna_logging)
        logging.getLogger("lightning.fabric").setLevel(self.lightning_fabric)
        logging.getLogger("lightning.pytorch").setLevel(self.lightning_pytorch)


class MyOptunaSearch(OptunaSearch):
    def __init__(self, func_kwargs: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func_kwargs = func_kwargs

    def _suggest_from_define_by_run_func(
        self,
        func: Callable[["OptunaTrial"], Optional[Dict[str, Any]]],
        ot_trial: "OptunaTrial",
    ) -> Dict:
        captor = _OptunaTrialSuggestCaptor(ot_trial)
        time_start = time.time()

        # Fetch all the trials to consider.
        # In this example, we use only completed trials, but users can specify other states
        # such as TrialState.PRUNED and TrialState.FAIL.
        states_to_consider = (
            TrialState.COMPLETE,
            TrialState.FAIL,
        )
        trials_to_consider = ot_trial.study.get_trials(
            deepcopy=False, states=states_to_consider
        )
        # Check whether we already evaluated the sampled `(x, y)`.
        for t in reversed(trials_to_consider):
            if ot_trial.params == t.params:
                # Use the existing value as trial duplicated the parameters.
                ret = {'score': t.value}
                break
        else:
            # noinspection PyArgumentList
            ret = func(captor, **self.func_kwargs)
        time_taken = time.time() - time_start
        if time_taken > DEFINE_BY_RUN_WARN_THRESHOLD_S:
            warnings.warn(
                "Define-by-run function passed in the `space` argument "
                f"took {time_taken} seconds to "
                "run. Ensure that actual computation, training takes "
                "place inside Tune's train functions or Trainables "
                "passed to `tune.Tuner()`."
            )
        if ret is not None:
            if not isinstance(ret, dict):
                raise TypeError(
                    "The return value of the define-by-run function "
                    "passed in the `space` argument should be "
                    "either None or a `dict` with `str` keys. "
                    f"Got {type(ret)}."
                )
            if not all(isinstance(k, str) for k in ret.keys()):
                raise TypeError(
                    "At least one of the keys in the dict returned by the "
                    "define-by-run function passed in the `space` argument "
                    "was not a `str`."
                )
        return {
            **captor.captured_values,
            **ret
        } if ret else captor.captured_values


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def optimize(
        self,
        search_space: Callable[[evaluation.Trial, AutoQMLFitCommand],
                               Configuration],
        X: InputData,
        y: TargetData,
        time_budget: timedelta,
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
        metric_: metric.Metric = metric.Accuracy(),
        seed: Union[int, None] = 0,
        sampler: Union[OptunaBaseSampler, None] = None,
        num_startup_trials: int = 100,
        time_budget_for_trials: Union[timedelta, None] = None,
        selection: str = "split"  # or cv
    ) -> Configuration:
        raise NotImplementedError()


def threaded_evaluation(
    result_queue, trial, X_train, y_train, X_test, y_test, metric_, selection,
    backend, pipeline_factory, logger
):
    """ Special function to run the evaluation in a separate thread to be able to terminate it if 
    it exceeds the time budget """

    try:
        loss = evaluation.evaluate(
            trial,
            X_train,
            y_train,
            X_test,
            y_test,
            metric=metric_,
            selection=selection,
            backend=backend,
            pipeline_factory=pipeline_factory,
        )
    except Exception:
        logger.error(msg='Failed to fit configuration', exc_info=True)
        loss = metric_.worst_result
    except RuntimeError as e:
        # Check if it's the specific error you want to handle
        if "Maximum number of iterations reached" in str(e):
            logger.error(
                msg=
                'Failed to fit configuration: Maximum number of iterations reached',
                exc_info=True
            )
            loss = metric_.worst_result
        else:
            raise  # Re-raise other RuntimeErrors

    result_queue.put(loss)


class RayOptimizer(Optimizer):
    def __init__(self, local_mode: bool = True):
        self.local_mode = local_mode

    def optimize(
        self,
        search_space: Callable[[evaluation.Trial, AutoQMLFitCommand],
                               Configuration],
        X: InputData,
        y: TargetData,
        time_budget: timedelta,
        fit_cmd: AutoQMLFitCommand,
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
        metric_: metric.Metric,
        seed: Union[int, None] = 0,
        sampler: Union[OptunaBaseSampler, None] = None,
        num_startup_trials: int = 100,
        time_budget_for_trials: Union[timedelta, None] = None,
        selection: str = "cv"  # or split
    ) -> Configuration:
        # 2. Split input data into train and test set
        if selection == "split":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=777
            )
        elif selection == "cv":
            X_train, X_test, y_train, y_test = X, None, y, None
        elif selection == "time_ordered":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
        else:
            raise ValueError(f"Selection method {selection} not supported")

        # Check if the search space is effectively static
        if is_single_configuration(search_space, {
            'cmd': fit_cmd,
            'pipeline_factory': pipeline_factory
        }):
            logger = logging.getLogger(__name__)
            logger.warning("Only one configuration found in the search space. Skipping tuning.")
            # Evaluate the single configuration once
            fixed_trial = evaluation.Trial(
                id='autoqml_fixed_trial_' + str(uuid.uuid4()),
                configuration=search_space(optuna.trial.FixedTrial({}), fit_cmd, pipeline_factory),
                loss=None,
                budget=None,
                duration=None
            )

            return fixed_trial.configuration

        output_control = OutputControl()

        if fit_cmd.verbosity < 2:
            original_stdout_fd = output_control.output_off()
        else:
            original_stdout_fd = copy.copy(os.dup(1))

        def _trainable(config: Configuration, verbosity: int, stdout_fd):
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            trial_uuid = 'autoqml_trial_' + str(uuid.uuid4())

            trial = evaluation.Trial(
                # previously: id=ray.tune.progress_reporter._get_trial_info()
                id=trial_uuid,
                configuration=config,
                loss=None,
                budget=None,
                duration=None
            )
            startt = time.time()

            if time_budget_for_trials:
                # Running model evaluation is a separate process to be able to terminate it if it exceeds the time budget
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=threaded_evaluation,
                    args=(
                        result_queue, trial, X_train, y_train, X_test, y_test,
                        metric_, selection, backend, pipeline_factory, logger
                    )
                )
                process.start()
                process.join(time_budget_for_trials.total_seconds())

                if process.is_alive():
                    logging.warning(
                        "Trial exceeded timeout and was terminated."
                    )
                    process.terminate()
                    process.join()  # Ensure resources are cleaned up
                    loss = metric_.worst_result
                else:
                    try:
                        loss = result_queue.get(
                        )  # Get the result without blocking
                    except queue.Empty:
                        logging.warning(
                            "Trial exceeded timeout and was terminated."
                        )
                        loss = metric_.worst_result

            else:
                result_queue = multiprocessing.Queue()
                threaded_evaluation(
                    result_queue, trial, X_train, y_train, X_test, y_test,
                    metric_, selection, backend, pipeline_factory, logger
                )
                try:
                    loss = result_queue.get(
                    )  # Get the result without blocking
                except queue.Empty:
                    logging.warning("Evaluation failed.")
                    loss = metric_.worst_result

            finisht = time.time()
            trial.duration = finisht - startt
            trial.loss = loss
            duration_td = datetime.fromtimestamp(
                finisht
            ) - datetime.fromtimestamp(startt)
            logger.info(
                f"Trial start: {datetime.fromtimestamp(startt)}, "
                f"Trial duration: {duration_td.total_seconds()}, "
                f"Trial loss: {loss}, "
                f"Configuration: {config}"
            )

            if verbosity == 1:
                method = config.get(
                    "regression__choice",
                    config.get("classification__choice", "")
                )

                s = (
                    f"Trial start: {datetime.fromtimestamp(startt)}, "
                    f"Trial duration: {duration_td.total_seconds()}, "
                    f"Trial loss: {loss}"
                )

                if method:
                    s += f", Method: {method}"

                os.write(stdout_fd, s.encode('utf-8') + b"\n")

            train.report({'score': trial.loss})
            return {'score': trial.loss}

        if sampler is None:
            sampler = TPESampler(
                n_startup_trials=
                num_startup_trials,  # Default value has to be discussed
                n_ei_candidates=24,
                multivariate=True,
                warn_independent_sampling=False,
                seed=seed
            )

        algo = MyOptunaSearch(
            func_kwargs={
                'cmd': fit_cmd,
                'pipeline_factory': pipeline_factory,
            },
            space=search_space,
            metric='score',
            mode='min' if metric_.mode_is_minimization else 'max',
            seed=seed if sampler is None else None,
            sampler=sampler,
        )

        ray.init(
            configure_logging=False,
            local_mode=self.local_mode,
            logging_level=0
        )
        tuner = tune.Tuner(
            partial(
                _trainable,
                verbosity=fit_cmd.verbosity,
                stdout_fd=original_stdout_fd
            ),
            tune_config=tune.TuneConfig(
                search_alg=algo,
                time_budget_s=time_budget,
                mode='min' if metric_.mode_is_minimization else 'max',
                metric='score',
                num_samples=-1,
            ),
            run_config=air.RunConfig(
                stop={'time_total_s': time_budget.total_seconds()}
            )
        )
        best_result = tuner.fit().get_best_result(filter_nan_and_inf=False)
        best_config: Optional[dict[str, Any]] = best_result.config
        best_score = best_result.metrics['score']
        ray.shutdown()

        # Restore output buffers and logger configurations
        if fit_cmd.verbosity < 2:
            output_control.output_on()

        if (
            metric_.mode_is_minimization and best_score >= metric_.worst_result
        ) or (
            not metric_.mode_is_minimization and
            best_score <= metric_.worst_result
        ):
            logger = logging.getLogger(__name__)
            logger.warning(
                msg="The optimization did not find a satisfactory configuration."
                "Consider rerunning or changing the search space."
            )
        return best_config
