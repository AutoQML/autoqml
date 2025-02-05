from autoqml.search_space.base import EstimatorChoice, TunableMixin
from autoqml.search_space.preprocessing.downsampling.resampling import Resampling
from autoqml.search_space.preprocessing.no_op import NoOp


class DownsamplingChoice(EstimatorChoice):
    @classmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'resampling': Resampling,
            'no-op': NoOp,
        }
