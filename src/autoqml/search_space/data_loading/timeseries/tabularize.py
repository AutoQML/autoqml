from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.base import BaseEstimator

from autoqml.constants import TargetData, InputData


def _calculate_cut_index(data_shape: Tuple[int, int], tile_size_x: int, tile_size_y: int) -> Tuple[int, int]:
    residue_x = data_shape[0] % tile_size_x
    residue_y = data_shape[1] % tile_size_y

    cut_index_x = data_shape[0] - residue_x
    cut_index_y = data_shape[1] - residue_y

    return cut_index_x, cut_index_y


def _separate_image_in_tiles(image: np.ndarray, tile_size_x: int, tile_size_y: int) -> np.ndarray:
    M = tile_size_x
    N = tile_size_y
    image_tiles = [image[x:x + M, y:y + N] for x in range(0, image.shape[0], M) for y in
                   range(0, image.shape[1], N)]

    image_tiles = np.array(image_tiles)

    return image_tiles.reshape(image_tiles.shape[0], image_tiles.shape[1] * image_tiles.shape[2])


class TabularizeTimeSeries(BaseEstimator):

    def __init__(self, tile_size_x: int, tile_size_y: int):
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y

    def transform(self, X: InputData, y: Union[pd.DataFrame, pd.Series]) -> Tuple[InputData, TargetData]:
        X = pd.DataFrame(X)
        y = y.values if isinstance(y, pd.Series) else y

        # It is probably better to paddle the image with zeros instead of cutting
        # Find good cut index for perfect separation of tiles
        cut_index_x, cut_index_y = _calculate_cut_index(X.shape, self.tile_size_x, self.tile_size_y)

        # Apply the cuts to the data and labels
        labels_reduced = y.values[:cut_index_x, :cut_index_y]

        # Separate the data and labels in tiles
        labels_tiles = _separate_image_in_tiles(labels_reduced, self.tile_size_x, self.tile_size_y)

        # Assign a label to each tile
        labels_mode, _ = mode(labels_tiles, keepdims=False, axis=1)

        training_reduced = X.iloc[:cut_index_x, :cut_index_y]

        # Separate the data and labels in tiles
        training_tiles = _separate_image_in_tiles(training_reduced.values, self.tile_size_x, self.tile_size_y)

        return pd.DataFrame(training_tiles), pd.Series(labels_mode)
