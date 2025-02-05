from autoqml.search_space.base import EstimatorChoice, TunableMixin
from autoqml.search_space.preprocessing.rescaling.standard_scaling import StandardScaling
from autoqml.search_space.preprocessing.rescaling.normalization import Normalization
from autoqml.search_space.preprocessing.rescaling.min_max_scaling import MinMaxScaling, MinMaxScalingForQuantumKernel
from autoqml.search_space.preprocessing.no_op import NoOp


class RescalingChoice(EstimatorChoice):
    @classmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'standard_scaling': StandardScaling,
            'normalization': Normalization,
            'min_max_scaling': MinMaxScaling,
            'no-op': NoOp,
        }


class RescalingChoiceQML(EstimatorChoice):
    @classmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'min_max_scaling': MinMaxScalingForQuantumKernel,
        }
