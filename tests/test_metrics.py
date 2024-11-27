# tests/test_metrics.py
import unittest
import numpy as np
import pandas as pd
from aryaxai_xai_metrics_tabular.explainer import SHAPExplainer, LIMEExplainer
from aryaxai_xai_metrics_tabular.metrics import ExplanationMetrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


class TestExplanationMetrics(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train = make_classification(n_samples=100, n_features=5, random_state=42)
        self.X_test, self.y_test = make_classification(n_samples=20, n_features=5, random_state=42)
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.features = [f"feature_{i}" for i in range(5)]
        self.shap_explainer = SHAPExplainer(self.model, features=self.features, X_train=self.X_train, task="classification")
        self.lime_explainer = LIMEExplainer(self.model, features=self.features, X_train=self.X_train, task="classification")

    def test_infidelity(self):
        metrics_calculator = ExplanationMetrics(
            model=self.model,
            explainer=self.shap_explainer,
            X_train=self.X_train,
            X_test=self.X_test,
            y_test=self.y_test,
            features=self.features,
            task="binary",
            metrics=['infidelity']
        )
        metrics_df = metrics_calculator.calculate_metrics()
        self.assertIn('infidelity', metrics_df.columns)

    def test_sensitivity(self):
        metrics_calculator = ExplanationMetrics(
            model=self.model,
            explainer=self.shap_explainer,
            X_train=self.X_train,
            X_test=self.X_test,
            y_test=self.y_test,
            features=self.features,
            task="binary",
            metrics=['sensitivity']
        )
        metrics_df = metrics_calculator.calculate_metrics()
        self.assertIn('sensitivity', metrics_df.columns)


if __name__ == '__main__':
    unittest.main()
