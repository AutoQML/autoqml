import abc
import random
import time
import copy
from datetime import datetime, timedelta
from typing import Callable

import sys
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
from optuna import Trial
from optuna.samplers import BaseSampler as OptunaBaseSampler
from sklearn.utils.validation import check_is_fitted

from autoqml.constants import InputData, TargetData
from autoqml.messages import AutoQMLFitCommand
from autoqml.meta_learning.datastatistics import TabularStatistics
from autoqml.optimizer import evaluation
from autoqml.optimizer.metric import Accuracy, BalancedAccuracy, Metric, RMSE
from autoqml.optimizer.optimizer import RayOptimizer
from autoqml.search_space import Configuration, SearchSpace
from autoqml.search_space.base import TunablePipeline
from autoqml.search_space.classification import ClassificationChoice
from autoqml.search_space.data_cleaning.imputation import ImputationChoice
from autoqml.search_space.preprocessing.dim_reduction import \
    DimReductionChoice
from autoqml.search_space.preprocessing.downsampling import \
    DownsamplingChoice
from autoqml.search_space.preprocessing.encoding import EncoderChoice
from autoqml.search_space.preprocessing.rescaling import RescalingChoice, RescalingChoiceQML
from autoqml.search_space.regression import RegressionChoice


class AutoQMLError(Exception):
    pass


class DataValidationError(AutoQMLError):
    pass


class EmptyPipelineError(RuntimeError):
    pass


def _define_by_run_func(
    trial: Trial,
    cmd: AutoQMLFitCommand,
    data_statistics: TabularStatistics,
    pipeline_factory: Callable[..., SearchSpace],
) -> Configuration:
    # 4.Construct search space
    pipeline = pipeline_factory()
    if pipeline is None:
        raise EmptyPipelineError(
            'The pipeline_factory() returned an empty pipeline. '
            'Please check the implementation of '
            'AutoQML._construct_search_space() for your machine learning task.'
        )

    # 4.1 Use Meta-Learning to Adjust search space with additional data properties
    # TODO

    # 4.2 Overwrite whatever config parameter the user specified, with the user specified value.
    defaults = cmd.configuration

    config = pipeline.sample_configuration(trial, defaults, data_statistics)
    return config


def _setup_logging(filename: str = None) -> None:
    """ Function to set up logging for the optimizer.

    Args:
        filename (str): The filename to save the log file to. If None, no log file will be created.
    """

    if filename is None:
        return None

    # Set up general logger for warnings and errors to the console
    logger = logging.getLogger("autoqml.optimizer.optimizer")

    # Remove all handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Clear all filters
    for filter in logger.filters:
        logger.removeFilter(filter)

    logger.setLevel(
        logging.DEBUG
    )  # Allows all levels to pass through the logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(
        logging.WARNING
    )  # Log only WARNING and ERROR to console
    logger.addHandler(console_handler)

    filename += "_" + str(
        datetime.fromtimestamp(time.time()).strftime(r"%Y-%m-%d_%H-%M-%S")
    ) + ".log"

    # Set up separate file handler for only INFO logs
    info_file_handler = RotatingFileHandler(filename)
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.addFilter(
        lambda record: record.levelno == logging.INFO
    )  # Only INFO level
    logger.addHandler(info_file_handler)


class AutoQML(abc.ABC):
    def __init__(self, seed: int = 42):
        self.seed = seed

    def _validate_input_parameters(self, cmd: AutoQMLFitCommand) -> None:

        all_qc_reg_methods = list(RegressionChoice.get_qc_components().keys())
        config_quantum_regression = {
            'autoqml.search_space.regression.RegressionChoice__choice':
                copy.copy(all_qc_reg_methods)
        }
        all_qc_reg_methods.remove("qnnr")
        config_quantum_regression_without_qnn = {
            'autoqml.search_space.regression.RegressionChoice__choice':
                copy.copy(all_qc_reg_methods)
        }
        config_classic_regression = {
            'autoqml.search_space.regression.RegressionChoice__choice':
                list(RegressionChoice.get_classic_components().keys())
        }

        all_qc_class_methods = list(
            ClassificationChoice.get_qc_components().keys()
        )
        config_quantum_classification = {
            'autoqml.search_space.classification.ClassificationChoice__choice':
                copy.copy(all_qc_class_methods)
        }
        all_qc_class_methods.remove("qnnc")
        config_quantum_classification_without_qnn = {
            'autoqml.search_space.classification.ClassificationChoice__choice':
                copy.copy(all_qc_class_methods)
        }
        config_classic_classification = {
            'autoqml.search_space.classification.ClassificationChoice__choice':
                list(ClassificationChoice.get_classic_components().keys())
        }

        if cmd.X is None:
            raise DataValidationError('X is None')
        if cmd.y is None:
            raise DataValidationError('y is None')
        if cmd.configuration is None:
            cmd.configuration = {}

        if isinstance(cmd.configuration, str):
            if cmd.configuration == "quantum_regression":
                cmd.configuration = config_quantum_regression
            elif cmd.configuration == "quantum_regression_without_qnn":
                cmd.configuration = config_quantum_regression_without_qnn
            elif cmd.configuration == "classic_regression":
                cmd.configuration = config_classic_regression
            elif cmd.configuration == "quantum_classification":
                cmd.configuration = config_quantum_classification
            elif cmd.configuration == "quantum_classification_without_qnn":
                cmd.configuration = config_quantum_classification_without_qnn
            elif cmd.configuration == "classic_classification":
                cmd.configuration = config_classic_classification
            else:
                raise DataValidationError(
                    'configuration is not a valid string.'
                )

        if not isinstance(cmd.X, InputData):
            raise DataValidationError(
                'X is not an instance of InputData: '
                'Not a Pandas DataFrame or NumPy ndarray.'
            )
        if not isinstance(cmd.y, TargetData):
            raise DataValidationError(
                'y is not an instance of TargetData: '
                'Not a Pandas Series or Numpy ndarray.'
            )
        if cmd.X.shape[0] == 0:
            raise DataValidationError('X is empty')
        if cmd.y.shape[0] == 0:
            raise DataValidationError('y is empty')
        if cmd.time_budget_for_this_task < timedelta(seconds=1):
            raise DataValidationError('time_left_for_this_task is too short.')

        if cmd.time_budget_for_trials is None:
            cmd.time_budget_for_trials = cmd.time_budget_for_this_task
        else:
            if cmd.time_budget_for_trials < timedelta(seconds=1):
                raise DataValidationError('time_left_for_trials is too short.')

        if cmd.sampler is not None and not isinstance(
            cmd.sampler, OptunaBaseSampler
        ):
            raise DataValidationError(
                'sampler is not an instance of OptunaBaseSampler.'
            )
        if cmd.num_startup_trials < 1:
            raise DataValidationError('num_startup_trials must be at least 1.')

        if cmd.selection not in ["cv", "split", "time_ordered"]:
            raise DataValidationError(
                'selection must be either "cv", "split" or "time_ordered".'
            )

    @abc.abstractmethod
    def _get_metric(self) -> Metric:
        raise NotImplementedError()

    def fit(
        self,
        cmd: AutoQMLFitCommand,
    ) -> 'AutoQML':
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 1. Validate input parameters
        self._validate_input_parameters(cmd)

        # 2. Create search space
        n_features = cmd.X.shape[1]
        n_samples = cmd.X.shape[0]

        # 3. Analyse input data to create Statistics
        data_statistics = TabularStatistics(
            n_samples=n_samples, n_features=n_features
        )

        # 5. Create Optimizer instance with adjusted search space
        optimizer = RayOptimizer()

        # 5.1 Setup logging
        _setup_logging(cmd.log_file)

        # 6. Execute optimization to find optimal configuration
        best_config = optimizer.optimize(
            _define_by_run_func,
            X=cmd.X,
            y=cmd.y,
            time_budget=cmd.time_budget_for_this_task,
            fit_cmd=cmd,
            backend=cmd.backend,
            data_statistics=data_statistics,
            pipeline_factory=self._construct_search_space,
            metric_=self._get_metric(),
            time_budget_for_trials=cmd.time_budget_for_trials,
            seed=cmd.seed,
            sampler=cmd.sampler,
            num_startup_trials=cmd.num_startup_trials,
            selection=cmd.selection
        )

        if not best_config:
            raise RuntimeError(
                "Could not find a valid configuration with the provided constraints."
            )

        # 7. Return fitted model instance from optimal configuration
        pipeline = evaluation.fit_configuration(
            X=cmd.X,
            y=cmd.y,
            config=best_config,
            backend=cmd.backend,
            pipeline_factory=self._construct_search_space,
            trial=None,
        )
        self.pipeline_ = pipeline

        return self

    def predict(self, X: InputData) -> TargetData:
        # 1. Check if already fitted
        check_is_fitted(self)

        return self.pipeline_.predict(X)

    @abc.abstractmethod
    def _construct_search_space(self) -> SearchSpace:
        raise NotImplementedError()


class AutoQMLTabularClassification(AutoQML):
    def _construct_search_space(self) -> TunablePipeline:
        # yapf: disable
        pipeline = TunablePipeline(
            steps=[
                ('encoding', EncoderChoice()),
                ('rescaling', RescalingChoice()),
                ('dim_reduction', DimReductionChoice()),
                ('rescalingQC', RescalingChoiceQML()),
                ('classification', ClassificationChoice()),
            ]
        )
        # yapf: enable
        return pipeline

    def _get_metric(self) -> Metric:
        return Accuracy()


class AutoQMLTimeSeriesClassification(AutoQML):
    def __init__(self, metric: str = "accuracy"):
        super().__init__()
        if metric not in ["accuracy", "balanced_accuracy"]:
            raise ValueError(
                "Invalid metric. Must be either 'accuracy' or 'balanced_accuracy'."
            )
        self._metric = metric

    def _construct_search_space(self) -> TunablePipeline:
        # yapf: disable
        pipeline = TunablePipeline(
            steps=[
                ('encoding', EncoderChoice()),
                ('imputation', ImputationChoice()),
                ('rescaling', RescalingChoice()),
                ('dim_reduction', DimReductionChoice()),
                ('rescalingQC', RescalingChoiceQML()),
                ('downsampling', DownsamplingChoice()),
                ('classification', ClassificationChoice()),
            ]
        )
        # yapf: enable
        return pipeline

    def _get_metric(self) -> Metric:
        if self._metric == "balanced_accuracy":
            return BalancedAccuracy()
        elif self._metric == "accuracy":
            return Accuracy()
        else:
            raise ValueError(
                "Invalid metric. Must be either 'accuracy' or 'balanced_accuracy'."
            )


class AutoQMLImageClassification(AutoQML):
    def _construct_search_space(self) -> TunablePipeline:
        raise NotImplementedError()

    def _get_metric(self) -> Metric:
        raise NotImplementedError()


class AutoQMLTabularRegression(AutoQML):
    def _construct_search_space(self) -> TunablePipeline:
        # yapf: disable
        pipeline = TunablePipeline(
            steps=[
                ('encoding', EncoderChoice()),
                ('rescaling', RescalingChoice()),
                ('dim_reduction', DimReductionChoice()),
                ('rescalingQC', RescalingChoiceQML()),
                ('regression', RegressionChoice()),
            ]
        )
        # yapf: enable
        return pipeline

    def _get_metric(self) -> Metric:
        return RMSE()
