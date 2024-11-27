# tests/test_explainer.py
import unittest
import numpy as np
from aryaxai_xai_metrics_tabular.explainer import SHAPExplainer, LIMEExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


class TestExplainers(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train = make_classification(n_samples=100, n_features=5, random_state=42)
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.features = [f"feature_{i}" for i in range(5)]

    def test_shap_explainer(self):
        shap_explainer = SHAPExplainer(self.model, self.features, X_train=self.X_train)
        explanation = shap_explainer.explain(self.X_train, instance_idx=0)
        self.assertTrue(len(explanation) > 0)

    def test_lime_explainer(self):
        lime_explainer = LIMEExplainer(self.model, self.features, X_train=self.X_train)
        explanation = lime_explainer.explain(self.X_train, instance_idx=0)
        self.assertTrue(len(explanation) > 0)


if __name__ == '__main__':
    unittest.main()
