from autoqml_lib.search_space.base import EstimatorChoice, TunableMixin
from autoqml_lib.search_space.preprocessing.dim_reduction.pca import PCA
from autoqml_lib.search_space.preprocessing.dim_reduction.tsne import TSNE
from autoqml_lib.search_space.preprocessing.dim_reduction.umap import UMAP
from autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder import Autoencoder

from autoqml_lib.search_space.preprocessing.no_op import NoOp


class DimReductionChoice(EstimatorChoice):
    @classmethod
    def get_components(cls) -> dict[str, type[TunableMixin]]:
        return {
            'pca': PCA,
            'tsne': TSNE,
            'umap': UMAP,
            'autoencoder': Autoencoder,
            'no-op': NoOp,
        }
