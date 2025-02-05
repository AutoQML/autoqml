from typing import Optional

from autoqml.constants import InputData, TargetData
from autoqml.search_space.base import EstimatorChoice, TunableMixin
from autoqml.search_space.regression.classic.decision_tree_regressor import \
    DecisionTreeRegressor
from autoqml.search_space.regression.classic.gaussian_process_regressor import \
    GaussianProcessRegressor
from autoqml.search_space.regression.classic.kernel_ridge import \
    KernelRidge
from autoqml.search_space.regression.classic.linear_regressor import \
    LinearRegressor
from autoqml.search_space.regression.classic.nnr import NNRegressor
from autoqml.search_space.regression.classic.random_forest_regressor import \
    RandomForestRegressor
from autoqml.search_space.regression.classic.svr import SVR
from autoqml.search_space.regression.quantum.qgpr import QGPR
from autoqml.search_space.regression.quantum.qkrr import QKRR
from autoqml.search_space.regression.quantum.qnnr import QNNRegressor
from autoqml.search_space.regression.quantum.qsvr import QSVR
from autoqml.search_space.regression.quantum.qrcr import QRCRegressor
from sklearn.exceptions import NotFittedError


class RegressionChoice(EstimatorChoice):
    def predict(self, X: InputData) -> TargetData:
        if self.estimator is None:
            raise NotFittedError()
        # noinspection PyUnresolvedReferences
        return self.estimator.predict(X)

    def get_available_components(
        self,
        default: str = None,
        include: list[str] = None,
        exclude: list[str] = None
    ) -> tuple[dict[str, type[TunableMixin]], Optional[str]]:
        if include is not None:
            if include[0] == 'classic':
                include = list(self.get_classic_components().keys())
            if include[0] == 'qc':
                include = list(self.get_qc_components().keys())

        return super().get_available_components(default, include, exclude)

    @classmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        return {**cls.get_classic_components(), **cls.get_qc_components()}

    @classmethod
    def get_classic_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'svr': SVR,
            'gaussian_process_regressor': GaussianProcessRegressor,
            'kernel_ridge': KernelRidge,
            'random_forest_regressor': RandomForestRegressor,
            'decision_tree_regressor': DecisionTreeRegressor,
            'linear_regressor': LinearRegressor,
            'nnr': NNRegressor,
        }

    @classmethod
    def get_qc_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'qgpr': QGPR,
            'qsvr': QSVR,
            'qkrr': QKRR,
            'qnnr': QNNRegressor,
            'qrcr': QRCRegressor,
        }
