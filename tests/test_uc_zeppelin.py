from datetime import timedelta

import numpy as np
from autoqml_lib.automl import AutoQMLTabularRegression
from autoqml_lib.messages import AutoQMLFitCommand
from autoqml_lib.use_cases.zeppelin import data_import
from sklearn.model_selection import train_test_split


def test_classic_svr_with_pca(use_cases_dir):
    X, y = data_import.load_data(
        dir=use_cases_dir / 'zeppelin/test_data/',
        file='Caterpillar-308-final-2022-06-17.csv',
        rows_limit=80,
    )
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, random_state=42)
    assert (all(X_train['extension'].unique() == ['E', 'D', 'E2', 'C']))
    # yapf: disable
    assert (all(X_train['location'].unique() == ['PL', 'SE', 'IT', 'GB',
                                                    'BE', 'FR', 'IE', 'DE',
                                                    'FI', 'GE', 'AT', 'RO',
                                                    'EE', 'ES', 'NL', 'CH',
                                                    'SK']))
    # yapf: enable
    custom_config = {
        'autoqml_lib.search_space.regression.RegressionChoice__choice':
            'svr',
        'autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice':
            'standard_scaling',  # 'min_max_scaling',
        'autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice':
            'one-hot',
        'autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories':
            17,
        'autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency':
            1,
        'autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice':
            'no-op',
        'autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice':
            'pca',  # 'pca' or 'autoencoder'
        'autoqml_lib.search_space.preprocessing.dim_reduction.pca.PCA__n_components':
            10,
        'autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice':
            'no-op',
    }
    automl = AutoQMLTabularRegression()
    cmd = AutoQMLFitCommand(
        X_train,
        y_train,
        timedelta(seconds=1),
        configuration=custom_config,
    )
    automl = automl.fit(cmd)
    y_pred = automl.predict(X_test)
    assert (y_pred.shape == (20, ))
    # yapf: disable
    y_pred_sample = np.array(
        [
            50070.76552498, 49967.48855368, 51029.39998682, 50475.55557173,
            51184.61901013, 51133.04052199, 50116.17246487, 50654.49118121,
            51133.37458118, 50903.47003064, 50811.23936587, 51193.99856662,
            51222.50555686, 50955.45281061, 50872.57107049, 50504.96343467,
            49858.86489991, 50336.63720054, 50067.65970944, 50332.70777413,
        ]
    )
    # yapf: enable
    assert (np.allclose(y_pred, y_pred_sample, atol=50_000, rtol=0.2))


def test_quantum_svr_with_pca(use_cases_dir):
    X, y = data_import.load_data(
        dir=use_cases_dir / 'zeppelin/test_data/',
        file='Caterpillar-308-final-2022-06-17.csv',
        rows_limit=80,
    )
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, random_state=42)
    assert (all(X_train['extension'].unique() == ['E', 'D', 'E2', 'C']))
    # yapf: disable
    assert (all(X_train['location'].unique() == ['PL', 'SE', 'IT', 'GB',
                                                    'BE', 'FR', 'IE', 'DE',
                                                    'FI', 'GE', 'AT', 'RO',
                                                    'EE', 'ES', 'NL', 'CH',
                                                    'SK']))
    # yapf: enable
    custom_config = {
        'autoqml_lib.search_space.regression.RegressionChoice__choice':
            'qsvr',
        'autoqml_lib.search_space.regression.quantum.qsvr.QSVR__num_qubits':
            4,
        'autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice':
            'standard_scaling',  # 'min_max_scaling',
        'autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice':
            'one-hot',
        'autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories':
            17,
        'autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency':
            1,
        'autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice':
            'no-op',
        'autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice':
            'pca',  # 'pca' or 'autoencoder'
        'autoqml_lib.search_space.preprocessing.dim_reduction.pca.PCA__n_components':
            4,
        'autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice':
            'no-op',
    }
    automl = AutoQMLTabularRegression()
    cmd = AutoQMLFitCommand(
        X_train,
        y_train,
        timedelta(seconds=1),
        configuration=custom_config,
    )
    automl = automl.fit(cmd)
    y_pred = automl.predict(X_test)
    assert (y_pred.shape == (20, ))
    # yapf: disable
    y_pred_sample = np.array(
        [
            50070.76552498, 49967.48855368, 51029.39998682, 50475.55557173,
            51184.61901013, 51133.04052199, 50116.17246487, 50654.49118121,
            51133.37458118, 50903.47003064, 50811.23936587, 51193.99856662,
            51222.50555686, 50955.45281061, 50872.57107049, 50504.96343467,
            49858.86489991, 50336.63720054, 50067.65970944, 50332.70777413,
        ]
    )
    # yapf: enable
    assert (np.allclose(y_pred, y_pred_sample, atol=10_000, rtol=0.2))


def test_quantum_svr_with_autoencoder(use_cases_dir):
    X, y = data_import.load_data(
        dir=use_cases_dir / 'zeppelin/test_data/',
        file='Caterpillar-308-final-2022-06-17.csv',
        rows_limit=80,
    )
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, random_state=42)
    assert (all(X_train['extension'].unique() == ['E', 'D', 'E2', 'C']))
    # yapf: disable
    assert (all(X_train['location'].unique() == ['PL', 'SE', 'IT', 'GB',
                                                    'BE', 'FR', 'IE', 'DE',
                                                    'FI', 'GE', 'AT', 'RO',
                                                    'EE', 'ES', 'NL', 'CH',
                                                    'SK']))
    # yapf: enable
    custom_config = {
        'autoqml_lib.search_space.regression.RegressionChoice__choice':
            'qsvr',
        'autoqml_lib.search_space.regression.quantum.qsvr.QSVR__num_qubits':
            4,
        'autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice':
            'standard_scaling',  # 'min_max_scaling',
        'autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice':
            'one-hot',
        'autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories':
            17,
        'autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency':
            1,
        'autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice':
            'autoencoder',
        'autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder.Autoencoder__latent_dim':
            4,
        'autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice':
            'no-op',
        'autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice':
            'no-op',
    }
    automl = AutoQMLTabularRegression()
    cmd = AutoQMLFitCommand(
        X_train,
        y_train,
        timedelta(seconds=5),
        configuration=custom_config,
    )
    automl = automl.fit(cmd)
    y_pred = automl.predict(X_test)
    assert (y_pred.shape == (20, ))
    # yapf: disable
    y_pred_sample = np.array(
        [
            50497.80535302, 50497.61971672, 50537.90684942, 50573.93449205,
            50598.81929275, 50578.90179867, 50673.76907143, 50539.3452145,
            50561.41800001, 50545.1129181, 50506.41698404, 50580.78273621,
            50654.33255061, 50742.36816784, 50497.81959164, 50523.62772769,
            50652.53315316, 50549.69281894, 50570.33754208, 50497.80535302
        ]
    )
    # yapf: enable
    assert (np.allclose(y_pred, y_pred_sample, atol=10_000, rtol=0.3))
