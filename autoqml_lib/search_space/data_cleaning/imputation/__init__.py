from autoqml_lib.search_space.base import EstimatorChoice, TunableMixin
from autoqml_lib.search_space.data_cleaning.imputation.constant import ConstantImputation
from autoqml_lib.search_space.data_cleaning.imputation.mean import MeanImputation
from autoqml_lib.search_space.preprocessing.no_op import NoOp


class ImputationChoice(EstimatorChoice):
    @classmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'constant': ConstantImputation,
            # 'drop': DropImputation, # Changes shape of input data, not compatible with sklearn interface
            'mean': MeanImputation,
            'no-op': NoOp,
        }
