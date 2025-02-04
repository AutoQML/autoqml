import abc
from dataclasses import dataclass

from autoqml.backend import Backend

# TODO: Remove?


@dataclass
class BackendFeatures:
    pass


class BackendAnalysis(abc.ABC):

    @abc.abstractmethod
    def analyse(self, backend: Backend) -> BackendFeatures:
        raise NotImplementedError()


class SimulatorBackendAnalysis(BackendAnalysis):

    def analyse(self, backend: Backend) -> BackendFeatures:
        # TODO Whatever is reasonable to do
        pass


class IBMBackendAnalysis(BackendAnalysis):

    def analyse(self, backend: Backend) -> BackendFeatures:
        # TODO Whatever is reasonable to do
        pass
