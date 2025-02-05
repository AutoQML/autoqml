from unittest import TestCase
from unittest.mock import MagicMock
from optuna import Trial
from autoqml.meta_learning.datastatistics import DataStatistics
from autoqml.search_space import Configuration
from autoqml.search_space.regression.quantum.qnnr import QNNRegressor
from autoqml.search_space.regression.quantum.qsvr import QSVR


class TestSampleConfiguration(TestCase):
    def test_qnn(self):
        qnn = QNNRegressor()
        trial = MagicMock(spec=Trial)
        defaults = Configuration()
        dataset_statistics = MagicMock(spec=DataStatistics)

        config = qnn.sample_configuration(trial, defaults, dataset_statistics)

        expected_params = [
            "num_qubits",
            "encoding_circuit",
            "observable",
            "num_repetitions",
            "learning_rate",
            "batch_size",
            "epochs",
            "shuffle",
            "variance",
            "parameter_seed",
        ]
        for param in expected_params:
            self.assertIn(param, config)

    def test_kernel(self):
        qsvr = QSVR()
        trial = MagicMock(spec=Trial)
        defaults = Configuration()
        dataset_statistics = MagicMock(spec=DataStatistics)

        trial.suggest_categorical.side_effect = lambda name, choices: choices[0
                                                                             ]

        config = qsvr.sample_configuration(trial, defaults, dataset_statistics)

        expected_params = [
            "num_qubits",
            "encoding_circuit",
            "quantum_kernel",
            "C",
        ]
        for param in expected_params:
            self.assertIn(param, config)
