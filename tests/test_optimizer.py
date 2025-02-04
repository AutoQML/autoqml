import logging
from typing import Callable
from autoqml.automl import AutoQMLTimeSeriesClassification
from autoqml.optimizer.metric import Accuracy
from autoqml.messages import AutoQMLFitCommand
from autoqml.meta_learning.datastatistics import TabularStatistics
from datetime import timedelta

from autoqml.optimizer.optimizer import RayOptimizer
from autoqml.search_space import Configuration, SearchSpace
from optuna import Trial
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)


# must be global
def search_space(
    trial: Trial,
    cmd: AutoQMLFitCommand,
    data_statistics: TabularStatistics,
    pipeline_factory: Callable[..., SearchSpace],
) -> Configuration:
    return {'a': 0, 'b': 0.001}


def test_ray_optimizer():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2)
    optimizer = RayOptimizer()
    time_budget = timedelta(seconds=2)
    cmd = AutoQMLFitCommand(
        X,
        y,
        time_budget_for_this_task=time_budget,
        configuration={},
        backend=None,
    )
    data_statistics = TabularStatistics(
        n_samples=cmd.X.shape[0], n_features=cmd.X.shape[1]
    )

    pipeline_factory = AutoQMLTimeSeriesClassification()

    best = optimizer.optimize(
        search_space,
        cmd.X,
        cmd.y,
        time_budget=cmd.time_budget_for_this_task,
        fit_cmd=cmd,
        backend=cmd.backend,
        data_statistics=data_statistics,
        pipeline_factory=pipeline_factory._construct_search_space(),
        metric_=pipeline_factory._get_metric()
    )
    assert ('a' in best)
    assert ('b' in best)
