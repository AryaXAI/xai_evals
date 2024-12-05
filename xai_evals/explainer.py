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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import ElasticNet
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class SHAPExplainer:
    def __init__(self, model, features, task="binary-classification", X_train=None,classification_threshold=0.5,subset_samples=False,subset_number=100):
        """
        Initialize SHAP Explainer with model, features, and training data.
        :param model: Trained model (e.g., LogisticRegression, RandomForest, etc.)
        :param features: List of feature names
        :param task: Either "classification" or "regression"
        :param X_train: Training data used for SHAP explainer initialization
        """
        self.model = model
        self.features_original = features
        self.features = features.str.replace(' ', '_')
        self.task = task
        self.subset_samples = subset_samples
        self.subset_number = subset_number
        self.shap_type = None
        if X_train is None and not hasattr(self.model, 'predict_proba'):
            raise ValueError("Training data (X_train) must be provided for SHAP explainer.")
        self.explainer = self._select_explainer(X_train)
        self.classification_threshold = classification_threshold
        
        

    def _select_explainer(self, X_train):
        #if not isinstance(X_train, np.ndarray):
        #X_train.columns = X_train.columns.str.replace(' ', '_')

        """Select the appropriate SHAP explainer based on model type."""
        if self.subset_samples:
            X_train_sample = shap.kmeans(X=X_train, k=self.subset_number).data
        else:
            pass
       
        if isinstance(self.model,(GradientBoostingClassifier)) and self.task == "multiclass-classification":
            raise ValueError("SHAP explanation doesnt support SHAP for multi-class classification")
        elif isinstance(self.model, (KMeans,NearestCentroid,BaggingClassifier,VotingClassifier)):
            raise ValueError("SHAP explanation not supported for the Model.")
        elif isinstance(self.model, (RidgeClassifier)):
            raise ValueError("Model does have predict probability hence it not support SHAP explanation.")           
        elif isinstance(self.model, (HistGradientBoostingClassifier,LGBMClassifier, CatBoostClassifier,RandomForestClassifier, DecisionTreeClassifier, xgb.XGBClassifier, GradientBoostingClassifier, ExtraTreesClassifier)):
            #AdaBoostClassifier,BaggingClassifier not supported by treeshap
            self.shap_type = "Tree"
            if self.subset_samples:
                return shap.TreeExplainer(self.model, X_train_sample)
            else:
                return shap.TreeExplainer(self.model, X_train)
        elif isinstance(self.model, nn.Module): 
            self.shap_type = "NN"
            return shap.DeepExplainer(self.model, torch.tensor(X_train.values).float()) 
        elif hasattr(self.model, 'coef_') or isinstance(self.model, (LogisticRegression,LogisticRegressionCV,ElasticNet)):
            self.shap_type = "LRegression"
            return shap.LinearExplainer(self.model, X_train)
        else:
            self.shap_type = "NOA"
            return shap.KernelExplainer(self._model_predict, X_train)

    def _model_predict(self, X):
        """Wrapper for model's prediction function to ensure compatibility with SHAP."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.features_original)
        return self.model.predict_proba(X)

    def explain(self, X_test, instance_idx=0):
        """
        Explain a specific instance using SHAP.
        :param X_test: Test dataset (as DataFrame or numpy array)
        :param instance_idx: Index of the instance to explain
        :return: DataFrame of feature attributions for the explained instance
        """
        X_test = pd.DataFrame(X_test, columns=self.features_original)
        x_instance = X_test.iloc[instance_idx:instance_idx+1]
        try:
            shap_values = self.explainer.shap_values(np.array(x_instance))
        except Exception as e:
        # Catch general exceptions and check for ExplainerError
            if "Additivity check failed" in str(e):
                print("ExplainerError encountered:", e)
                print("Retrying with additivity check disabled...")
                
                # Retry with check_additivity=False
                shap_values =  self.explainer.shap_values(np.array(x_instance),check_additivity=False)
                print("SHAP values computed with additivity check disabled!")
            else:
                # Re-raise the exception if it's not related to the additivity check
                raise
        attributions = shap_values
        #print(self.task,self.shap_type,attributions.shape)
        if self.task == "binary-classification":
            if len(attributions.shape) == 3:
                idx = np.argmax(self._model_predict(x_instance))
                attributions = attributions[:,:,idx]
        elif self.task == "multiclass-classification":
            if len(attributions.shape) == 3:
                idx = np.argmax(self._model_predict(x_instance))
                attributions = attributions[:,:,idx]
        elif self.task == "multi-label-classification":
            pass
        else:
            pass
        #print(attributions.shape)
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
    def __init__(self, model, features, task="binary-classification", X_train=None,model_classes=None):
        """
        Initialize LIME Explainer with model, features, and training data.
        :param model: Trained model (e.g., LogisticRegression, RandomForest, etc.)
        :param features: List of feature names
        :param task: Either "classification" or "regression"
        :param X_train: Training data used for LIME explainer initialization
        """
        self.model = model
        self.features = features
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train
        elif isinstance(X_train, np.ndarray):
            self.X_train = pd.DataFrame(X_train, columns=self.features)

        self.features = [feature.replace(' ', '_') for feature in self.features]
        self.task = task
        self.model_classes = model_classes
        self.categorical_features = self._identify_categorical_features()
        self.X_train = self.X_train.to_numpy()
        # Identify categorical features based on dtype
        if self.task == "Regression" or self.task == "regression":
            self.shap_task = "regression"
        else:
            self.shap_task = "classification"

        if X_train is None:
            raise ValueError("Training data (X_train) must be provided for LIME explainer.")

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.features,
            class_names=self.model_classes,
            categorical_features=self.categorical_features,
            mode=self.shap_task
        )
    
    def _identify_categorical_features(self):
        """
        Identifies categorical features based on their dtype.
        Assumes that categorical features are of type 'object' or 'category'.
        Additionally considers numeric columns with a low number of unique values as categorical.
        """
        categorical_features = []

        # Identify categorical features based on dtype
        for i, dtype in enumerate(self.X_train.dtypes):
            if dtype == 'object' or dtype.name == 'category':  # Traditional categorical types
                categorical_features.append(i)
            elif dtype in ['int64', 'float64']:  # Numeric types
                # Check if the number of unique values is small, indicating it might be categorical
                if len(self.X_train.iloc[:, i].unique()) < 10:
                    categorical_features.append(i)

        return categorical_features

    def explain(self, X_test, instance_idx=0):
        """
        Explain a specific instance using LIME.
        :param X_test: Test dataset (as DataFrame or numpy array)
        :param instance_idx: Index of the instance to explain
        :return: DataFrame of feature attributions for the explained instance
        """
        if isinstance(X_test, pd.DataFrame):
            self.X_test = X_test.to_numpy()
        elif isinstance(X_test, np.ndarray):
            self.X_test = X_test

        X_test = pd.DataFrame(self.X_test, columns=self.features)
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
