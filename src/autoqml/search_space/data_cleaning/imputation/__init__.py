from autoqml.search_space.base import EstimatorChoice, TunableMixin
from autoqml.search_space.data_cleaning.imputation.constant import ConstantImputation
from autoqml.search_space.data_cleaning.imputation.mean import MeanImputation
from autoqml.search_space.preprocessing.no_op import NoOp


class ImputationChoice(EstimatorChoice):
    @classmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'constant': ConstantImputation,
            # 'drop': DropImputation, # Changes shape of input data, not compatible with sklearn interface
            'mean': MeanImputation,
            'no-op': NoOp,
        }
