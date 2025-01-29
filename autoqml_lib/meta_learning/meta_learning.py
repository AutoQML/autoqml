from autoqml_lib.backend.analysis import BackendFeatures
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import SearchSpace


class MetaLearner:

    def adapt_search_space(
            self,
            search_space: SearchSpace,
            dataset_statistics: DataStatistics,
            backend_features: BackendFeatures
    ) -> SearchSpace:
        # 1. Prune available algorithms
        # 2. Prune available hyperparameters
        pass
