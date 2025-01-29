from typing import Union

import numpy as np
import pandas as pd

InputData = Union[pd.DataFrame, np.ndarray]
TargetData = Union[pd.Series, np.ndarray]

Budget = float

TrialId = str
