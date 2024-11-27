# shap_lime_explainer/explainer.py
import shap
import lime.lime_tabular
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import torch.nn as nn


class SHAPExplainer:
    def __init__(self, model, features, task="classification", X_train=None):
        """
        Initialize SHAP Explainer with model, features, and training data.
        :param model: Trained model (e.g., LogisticRegression, RandomForest, etc.)
        :param features: List of feature names
        :param task: Either "classification" or "regression"
        :param X_train: Training data used for SHAP explainer initialization
        """
        self.model = model
        self.features = features
        self.task = task
        if X_train is None and not hasattr(self.model, 'predict_proba'):
            raise ValueError("Training data (X_train) must be provided for SHAP explainer.")
        self.explainer = self._select_explainer(X_train)

    def _select_explainer(self, X_train):
        """Select the appropriate SHAP explainer based on model type."""
        if isinstance(self.model, (RandomForestClassifier, xgb.XGBClassifier)): 
            X_train = shap.kmeans(X_train, 10).data
            return shap.TreeExplainer(self.model, X_train)
        elif isinstance(self.model, nn.Module): 
            return shap.DeepExplainer(self.model, torch.tensor(X_train.values).float()) 
        elif hasattr(self.model, 'coef_') or isinstance(self.model, LogisticRegression):
            return shap.LinearExplainer(self.model, X_train)
        else:
            return shap.KernelExplainer(self._model_predict, X_train)

    def _model_predict(self, X):
        """Wrapper for model's prediction function to ensure compatibility with SHAP."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.features)
        return self.model.predict_proba(X)

    def explain(self, X_test, instance_idx=0):
        """
        Explain a specific instance using SHAP.
        :param X_test: Test dataset (as DataFrame or numpy array)
        :param instance_idx: Index of the instance to explain
        :return: DataFrame of feature attributions for the explained instance
        """
        X_test = pd.DataFrame(X_test, columns=self.features)
        x_instance = X_test.iloc[instance_idx:instance_idx+1]
        shap_values = self.explainer.shap_values(x_instance)
        attributions = shap_values
        return self._format_attributions(attributions, x_instance)

    def _format_attributions(self, attributions, x_instance):
        """Format SHAP attributions into a DataFrame."""
        attributions = attributions.flatten()
        feature_values = x_instance.values.flatten()
        attribution_df = pd.DataFrame({
            'Feature': self.features,
            'Value': feature_values,
            'Attribution': attributions
        })
        attribution_df = attribution_df.sort_values(by="Attribution", key=abs, ascending=False)
        return attribution_df


class LIMEExplainer:
    def __init__(self, model, features, task="classification", X_train=None):
        """
        Initialize LIME Explainer with model, features, and training data.
        :param model: Trained model (e.g., LogisticRegression, RandomForest, etc.)
        :param features: List of feature names
        :param task: Either "classification" or "regression"
        :param X_train: Training data used for LIME explainer initialization
        """
        self.model = model
        self.features = features
        self.task = task
        if X_train is None:
            raise ValueError("Training data (X_train) must be provided for LIME explainer.")
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.features,
            class_names=model.classes_ if self.task == "classification" else None,
            mode=self.task
        )

    def explain(self, X_test, instance_idx=0):
        """
        Explain a specific instance using LIME.
        :param X_test: Test dataset (as DataFrame or numpy array)
        :param instance_idx: Index of the instance to explain
        :return: DataFrame of feature attributions for the explained instance
        """
        X_test = pd.DataFrame(X_test, columns=self.features)
        x_instance = X_test.iloc[instance_idx:instance_idx+1]
        explanation = self.explainer.explain_instance(
            X_test.iloc[instance_idx].values, self.model.predict_proba
        )
        return self._map_binned_to_original(explanation.as_list(), x_instance)

    def _map_binned_to_original(self, attributions, x_instance):
        """Map LIME's binned features back to original features."""
        original_attributions = []
        for feature, attribution in attributions:
            if "<=" in feature or "<" in feature or ">" in feature:
                original_feature = next(word.strip() for word in feature.split() if word.strip() in self.features)
            else:
                original_feature = feature
            feature_value = x_instance[original_feature].values[0]
            original_attributions.append((original_feature, feature_value, attribution))
        attribution_df = pd.DataFrame(original_attributions, columns=['Feature', 'Value', 'Attribution'])
        attribution_df['Attribution'] = attribution_df['Attribution'].abs()
        return attribution_df
