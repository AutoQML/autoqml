from unittest import TestCase

from datetime import timedelta
from sklearn.datasets import make_classification

from autoqml import TimeSeriesClassification
from autoqml.messages import AutoQMLFitCommand


def _fit(X, y, custom_config: dict = dict()):
    automl = TimeSeriesClassification()
    cmd = AutoQMLFitCommand(
        X, y, timedelta(seconds=5), configuration=custom_config
    )
    automl.fit(cmd)
    return automl


class TestSearchSpaceClassic(TestCase):
    def test_default_fit(self):
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        automl = _fit(X, y)
        self.assertIsNotNone(automl.pipeline_)

    def test_custom_config_fit(self):
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        automl = _fit(
            X, y, {
                'autoqml.search_space.classification.ClassificationChoice__choice':
                    'svc'
            }
        )

        self.assertIsNotNone(automl.pipeline_)
        self.assertEqual(
            str(
                type(automl.pipeline_.named_steps['classification'].estimator)
            ), "<class 'autoqml.search_space.classification.classic.svc.SVC'>"
        )

    def test_custom_config_fit_predict(self):
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        automl = _fit(
            X, y, {
                'autoqml.search_space.classification.ClassificationChoice__choice':
                    'svc'
            }
        )
        self.assertIsNotNone(automl.pipeline_)

        y_pred = automl.predict(X)
        self.assertIsNotNone(y_pred)
        self.assertEqual(y_pred.shape, y.shape)


class TestSearchSpaceQuantum(TestCase):
    def test_qc_fit(self):
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
        automl = _fit(
            X, y, {
                'autoqml.search_space.classification.ClassificationChoice__choice':
                    'qsvc'
            }
        )

        self.assertIsNotNone(automl.pipeline_)
        self.assertEqual(
            str(
                type(automl.pipeline_.named_steps['classification'].estimator)
            ),
            "<class 'autoqml.search_space.classification.quantum.qsvc.QSVC'>"
        )

    def test_qc_fit_predict(self):
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
        automl = _fit(
            X, y, {
                'autoqml.search_space.classification.ClassificationChoice__choice':
                    'qsvc'
            }
        )
        self.assertIsNotNone(automl.pipeline_)

        y_pred = automl.predict(X)
        self.assertIsNotNone(y_pred)
        self.assertEqual(y_pred.shape, y.shape)
