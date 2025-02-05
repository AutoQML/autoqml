from typing import Optional

from sklearn.exceptions import NotFittedError

from autoqml.constants import InputData, TargetData
from autoqml.search_space.base import EstimatorChoice, TunableMixin
from autoqml.search_space.classification.quantum.qgpc import QGPC
from autoqml.search_space.classification.quantum.qnnc import QNNClassifier
from autoqml.search_space.classification.quantum.qsvc import QSVC
from autoqml.search_space.classification.quantum.qrcc import QRCClassifier
from autoqml.search_space.classification.classic.decision_tree_classifier import DecisionTreeClassifier
from autoqml.search_space.classification.classic.logistic_regression_classifier import LogisticRegressor
from autoqml.search_space.classification.classic.random_forest_classifier import RandomForestClassifier
from autoqml.search_space.classification.classic.svc import SVC
from autoqml.search_space.classification.classic.gaussian_process_classifier import GaussianProcessClassifier
from autoqml.search_space.classification.classic.ridge_classifier import RidgeClassifier
from autoqml.search_space.classification.classic.perceptron import Perceptron


class ClassificationChoice(EstimatorChoice):
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
            'random_forest_classifier': RandomForestClassifier,
            'logistic_regression_classifier': LogisticRegressor,
            'decision_tree_classifier': DecisionTreeClassifier,
            'svc': SVC,
            'gaussian_process_classifier': GaussianProcessClassifier,
            'ridge_classifier': RidgeClassifier,
            'perceptron': Perceptron
        }

    @classmethod
    def get_qc_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'qsvc': QSVC,
            'qgpc': QGPC,
            'qnnc': QNNClassifier,
            'qrcc': QRCClassifier,
        }
