from typing import Tuple

import numpy as np
import pandas as pd


def split_types(X) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(X)
    num_cols = df.select_dtypes(
        include=[
            'number',
            'bool_',
        ], exclude=['object_']
    ).columns.values
    cat_cols = df.select_dtypes(
        include=['object_'], exclude=[
            'number',
            'bool_',
        ]
    ).columns.values

    return num_cols, cat_cols
