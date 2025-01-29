import logging
from datetime import timedelta

import dill
import numpy as np
import pandas as pd
from autoqml_lib.automl import AutoQMLTimeSeriesClassification
from autoqml_lib.messages import AutoQMLFitCommand
from autoqml_lib.search_space.data_loading.timeseries.tabularize import \
    TabularizeTimeSeries
from autoqml_lib.use_cases import keb
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_pipeline_construction_classic() -> None:
    X, y = keb.load_data()
    X2: pd.DataFrame
    y2: pd.Series
    X2, y2 = TabularizeTimeSeries(7, 30).transform(X, y)

    X3: np.ndarray
    y3: np.ndarray
    X3, y3 = keb.reduce_zeroes(X=X2, y=y2)

    X_train, X_test, y_train, y_test = train_test_split(
        X3,
        y3,
        test_size=0.25,
        random_state=777,
        stratify=y3,
    )
    custom_config = {
        'autoqml_lib.search_space.classification.ClassificationChoice__choice':
            'svc',
        'autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice':
            'min_max_scaling',
        'autoqml_lib.search_space.preprocessing.encoding.EncodingChoice__choice':
            'no-op',
        'autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice':
            'no-op',
        'autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice':
            'no-op',
        'autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice':
            'no-op',
    }
    automl = AutoQMLTimeSeriesClassification()
    cmd = AutoQMLFitCommand(
        X_train,
        y_train,
        timedelta(seconds=1),
        configuration=custom_config,
    )
    automl = automl.fit(cmd)

    assert automl.pipeline_ is not None


def test_pipeline_results_classic(use_cases_dir) -> None:
    X, y = keb.load_data()
    X2: pd.DataFrame
    y2: pd.Series
    X2, y2 = TabularizeTimeSeries(7, 30).transform(X, y)

    X3: np.ndarray
    y3: np.ndarray
    X3, y3 = keb.reduce_zeroes(X=X2, y=y2)

    X_train, X_test, y_train, y_test = train_test_split(
        X3,
        y3,
        test_size=0.25,
        random_state=777,
        stratify=y3,
    )
    with open(
        use_cases_dir / 'keb/automl-fixed-parameters-classic.dill', 'rb'
    ) as file:
        automl = dill.load(file=file)

    y_pred = automl.predict(X_test)
    # yapf: disable
    test_array = np.array(
        [
            [35,  0,  2,  0,  0,  0,  1],
            [ 0, 70,  0,  0,  0,  0,  0],
            [ 0,  0, 31,  1,  0,  0,  1],
            [ 1,  0,  1,  6,  0,  0,  0],
            [ 1,  0,  0,  0,  5,  0,  0],
            [ 0,  0,  1,  0,  0, 57,  0],
            [ 3,  0,  0,  0,  0,  0, 55],
        ]
    )
    # yapf: enable
    cm_q = confusion_matrix(
        y_test,
        y_pred,
    )
    compare_res_arr = cm_q == test_array

    assert (compare_res_arr.all())

    assert accuracy_score(y_test, y_pred) == 0.955719557195572
    assert balanced_accuracy_score(y_test, y_pred) == 0.9106877695806915


def test_pipeline_construction_quantum() -> None:
    X, y = keb.load_data()
    X2: pd.DataFrame
    y2: pd.Series
    X2, y2 = TabularizeTimeSeries(7, 30).transform(X, y)

    X3: np.ndarray
    y3: np.ndarray
    X3, y3 = keb.reduce_zeroes(X=X2, y=y2)

    X_train, X_test, y_train, y_test = train_test_split(
        X3,
        y3,
        test_size=0.25,
        random_state=777,
        stratify=y3,
    )
    custom_config = {
        'autoqml_lib.search_space.classification.ClassificationChoice__choice':
            'qsvc',
        'autoqml_lib.search_space.classification.quantum.qsvc.QSVC__num_qubits':
            4,
        'autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice':
            'min_max_scaling',
        'autoqml_lib.search_space.preprocessing.encoding.EncodingChoice__choice':
            'no-op',
        'autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice':
            'no-op',
        'autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice':
            'pca',
        'autoqml_lib.search_space.preprocessing.dim_reduction.pca.PCA__n_components':
            4,
        'autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice':
            'no-op',
    }
    automl = AutoQMLTimeSeriesClassification()
    cmd = AutoQMLFitCommand(
        X_train,
        y_train,
        timedelta(seconds=1),
        configuration=custom_config,
    )
    automl = automl.fit(cmd)

    assert automl.pipeline_ is not None
