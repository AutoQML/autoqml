import pandas as pd

from autoqml_lib.use_cases import keb
from autoqml_lib.search_space.data_loading.timeseries.tabularize import TabularizeTimeSeries


def test_data_tabularize():
    X, y = keb.load_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)

    X2, y2 = TabularizeTimeSeries(7, 30).transform(X, y)
    assert isinstance(X2, pd.DataFrame)
    assert isinstance(y2, pd.Series)
