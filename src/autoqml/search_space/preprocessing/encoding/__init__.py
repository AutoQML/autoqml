from autoqml.search_space.base import EstimatorChoice, TunableMixin
from autoqml.search_space.preprocessing.encoding.categorical import CategoricalEncoder
from autoqml.search_space.preprocessing.encoding.one_hot import OneHotEncoder
from autoqml.search_space.preprocessing.no_op import NoOp


class EncoderChoice(EstimatorChoice):
    @classmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'categorical': CategoricalEncoder,
            'one-hot': OneHotEncoder,
            'no-op': NoOp,
        }
