import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xai_evals.explainer import LIMEExplainer,SHAPExplainer

class ExplanationMetrics:
    def __init__(self, model, explainer_name, X_train, X_test, y_test, features, task, metrics=None, start_idx=0, end_idx=None):
        """
        Initialize the ExplanationMetrics class.
        :param model: Trained model (e.g., XGBoost)
        :param explainer_name: Name of the explanation method ('shap' or 'lime')
        :param X_train: Training data
        :param X_test: Test data
        :param y_test: True labels of test data
        :param features: List of feature names
        :param task: Task type ('binary', 'multiclass', 'multilabel', or 'regression')
        :param metrics: List of metrics to calculate
        :param start_idx: Starting index of the dataset to evaluate
        :param end_idx: Ending index of the dataset to evaluate
        """
        self.model = model
        self.explainer_name = explainer_name
        self.X_train = X_train
        if isinstance(X_test, pd.DataFrame):
            self.X_test = X_test.to_numpy()
        elif isinstance(X_test, np.ndarray):
            self.X_test = X_test
        self.y_test = y_test
        self.features = features
        self.task = task
        self.metrics = metrics if metrics else ['faithfulness', 'infidelity', 'sensitivity', 'comprehensiveness', 'sufficiency', 'monotonicity', 'complexity', 'sparseness']
        #'auc_tp']
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx else len(X_test)

        # Initialize explainer based on the name
        self.explainer = self._initialize_explainer()

    def _initialize_explainer(self):
        """
        Initialize the explainer based on the explainer_name ('shap' or 'lime').
        :return: Corresponding explainer instance (LIMEExplainer or SHAPExplainer)
        """
        if self.explainer_name == 'shap':
            return SHAPExplainer(model=self.model, features=self.features, task=self.task, X_train=self.X_train)
        elif self.explainer_name == 'lime':
            if self.task == 'binary':
                task = 'classification'
            else:
                task = self.task
            return LIMEExplainer(model=self.model, features=self.features, task=task, X_train=self.X_train)
        else:
            raise ValueError("Unsupported explainer name. Choose 'shap' or 'lime'.")

    def _get_explanation(self, start_idx, end_idx):
        """
        Generate explanation using the explainer for the given method ('shap' or 'lime') for multiple instances.
        :return: List of DataFrames for each instance's attributions.
        """
        attributions_list = []
        for i in range(start_idx, end_idx):
            instance_explanation = self.explainer.explain(self.X_test, instance_idx=i)
            attributions_list.append(instance_explanation)  # Store each instance's attribution as a DataFrame
        return attributions_list

    def _generate_perturbations(self, X, epsilon=1e-2):
        """
        Generate perturbations by adding noise to the features.
        :param X: The input dataset
        :param epsilon: The noise factor (controls the size of perturbations)
        :return: Perturbed instances
        """
        return X + np.random.normal(0, epsilon, size=X.shape)

    def _infidelity(self, attributions_list):
        """
        Calculate the infidelity metric.
        """
        infidelity_scores = []
        perturbations = self._generate_perturbations(self.X_test)  # Generate perturbations for all test instances
        for i in range(self.start_idx, self.end_idx):
            perturbation = perturbations[i]
            predicted_impact = np.dot(perturbation, attributions_list[i]['Attribution'].values)
            actual_impact = self.model.predict_proba((self.X_test[i] + perturbation).reshape(1, -1))[0, 1] - \
                            self.model.predict_proba(self.X_test[i].reshape(1, -1))[0, 1]
            infidelity_scores.append((predicted_impact - actual_impact)**2)
        return np.mean(infidelity_scores)

    def _sensitivity(self, attributions_list, epsilon=1e-2):
        """
        Calculate the sensitivity metric.
        """
        sensitivity_scores = []
        for i in range(self.start_idx, self.end_idx):
            perturbed_instance = self.X_test[i] + np.random.normal(0, epsilon, size=self.X_test[i].shape)
            new_attributions = self.model.predict_proba(perturbed_instance.reshape(1, -1))[0, 1]
            sensitivity_scores.append(np.linalg.norm(attributions_list[i]['Attribution'].values - new_attributions))
        return np.mean(sensitivity_scores)

    def _comprehensiveness(self, attributions_list, k=5):
        """
        Calculate the comprehensiveness metric.
        """
        top_k_indices = np.argsort([attr['Attribution'].values for attr in attributions_list], axis=1)[:, -k:]
        comprehensiveness_scores = []
        for i in range(self.start_idx, self.end_idx):
            masked_instance = self.X_test[i].copy()
            masked_instance[top_k_indices[i]] = 0
            baseline_prediction = self.model.predict_proba(self.X_test[i:i+1])[0, 1]
            masked_prediction = self.model.predict_proba(masked_instance.reshape(1, -1))[0, 1]
            comprehensiveness_scores.append(baseline_prediction - masked_prediction)
        return np.mean(comprehensiveness_scores)

    def _sufficiency(self, attributions_list, k=5):
        """
        Calculate the sufficiency metric.
        """
        top_k_indices = np.argsort([attr['Attribution'].values for attr in attributions_list], axis=1)[:, -k:]
        sufficiency_scores = []
        for i in range(self.start_idx, self.end_idx):
            focused_instance = np.zeros_like(self.X_test[i])
            focused_instance[top_k_indices[i]] = self.X_test[i][top_k_indices[i]]
            baseline_prediction = self.model.predict_proba(self.X_test[i:i+1])[0, 1]
            focused_prediction = self.model.predict_proba(focused_instance.reshape(1, -1))[0, 1]
            sufficiency_scores.append(focused_prediction - baseline_prediction)
        return np.mean(sufficiency_scores)

    def _monotonicity(self, attributions_list):
        """
        Calculate the monotonicity metric.
        """
        monotonicity_scores = []
        for attributions in attributions_list:
            sorted_attr = np.sort(attributions['Attribution'].values)
            monotonicity_scores.append(np.all(np.diff(sorted_attr) >= 0))
        return np.mean(monotonicity_scores)

    def _auc_tp(self, attributions_list, k=10):
        """
        AUC calculation for the top-k features
        Handles binary, multi-class, and multi-label classification.
        """
        auc_scores = []

        if self.task == 'binary':
            # Binary classification: Calculate AUC for each instance
            for i in range(self.start_idx, self.end_idx):
                # Check if y_true contains both 0 and 1
                if len(np.unique(self.y_test[i])) == 1:  # Only one class present
                    auc_scores.append(np.nan)  # Return NaN if there's only one class
                else:
                    sorted_indices = np.argsort(attributions_list[i]['Attribution'].values)[-k:]
                    auc_scores.append(roc_auc_score(np.array([self.y_test[i]]), attributions_list[i]['Attribution'].values[sorted_indices]))

        elif self.task == 'multiclass':
            # Multi-class classification: Calculate AUC for multi-class classification
            for i in range(self.start_idx, self.end_idx):
                sorted_indices = np.argsort(attributions_list[i]['Attribution'].values)[-k:]
                auc_scores.append(roc_auc_score(self.y_test[i], attributions_list[i]['Attribution'].values[sorted_indices], multi_class='ovr'))

        elif self.task == 'multilabel':
            # Multi-label classification: Calculate AUC for each label (for multi-label, AUC is calculated for each label separately)
            for i in range(self.start_idx, self.end_idx):
                sorted_indices = np.argsort(attributions_list[i]['Attribution'].values)[-k:]
                auc_scores.append(roc_auc_score(self.y_test[i], attributions_list[i]['Attribution'].values[sorted_indices], average='macro'))

        elif self.task == 'regression':
            # Regression: Use AUC for regression is less common, but we could compute R-squared or other metrics instead.
            pass

        return np.nanmean(auc_scores)  # Use np.nanmean to handle NaN values properly

    def _complexity(self, attributions_list):
        """
        Calculate the complexity metric.
        """
        return np.mean([np.sum(attr['Attribution'].values != 0) for attr in attributions_list])

    def _sparseness(self, attributions_list):
        """
        Calculate the sparseness metric.
        """
        return 1 - (np.sum([np.sum(attr['Attribution'].values != 0) for attr in attributions_list]) / 
                    (len(attributions_list) * len(self.features)))
                    
    def _faithfulness_correlation(self, attributions_list):
        # Faithfulness calculation
        faithfulness_scores = []
        for i in range(self.start_idx, self.end_idx):
            baseline_prediction = self.model.predict_proba(self.X_test[i:i+1])[0, 1]

            # Access the attributions for the current instance from the list of DataFrames
            attribution_values = attributions_list[i]['Attribution'].values

            for j, attribution in enumerate(attribution_values):
                perturbed_instance = self.X_test[i].copy()
                perturbed_instance[j] = 0
                perturbed_prediction = self.model.predict_proba(perturbed_instance.reshape(1, -1))[0, 1]
                change = baseline_prediction - perturbed_prediction
                faithfulness_scores.append((change, attribution))

        changes, attrs = zip(*faithfulness_scores)
        return np.corrcoef(changes, attrs)[0, 1]

    def calculate_metrics(self):
        """
        Calculate the specified metrics and return the results as a DataFrame.
        :return: DataFrame with metrics and their values.
        """
        # Generate explanations for the entire dataset or the specified range
        attributions_list = self._get_explanation(self.start_idx, self.end_idx)

        # Initialize result dictionary
        results = {}

        # Calculate requested metrics
        if 'faithfulness' in self.metrics:
            results['faithfulness'] = self._faithfulness_correlation(attributions_list)
        if 'infidelity' in self.metrics:
            results['infidelity'] = self._infidelity(attributions_list)
        if 'sensitivity' in self.metrics:
            results['sensitivity'] = self._sensitivity(attributions_list)
        if 'comprehensiveness' in self.metrics:
            results['comprehensiveness'] = self._comprehensiveness(attributions_list)
        if 'sufficiency' in self.metrics:
            results['sufficiency'] = self._sufficiency(attributions_list)
        if 'monotonicity' in self.metrics:
            results['monotonicity'] = self._monotonicity(attributions_list)
        if 'auc_tp' in self.metrics:
            results['auc_tp'] = self._auc_tp(attributions_list)
        if 'complexity' in self.metrics:
            results['complexity'] = self._complexity(attributions_list)
        if 'sparseness' in self.metrics:
            results['sparseness'] = self._sparseness(attributions_list)

        # Return the results as a DataFrame
        return pd.DataFrame(results, index=[0])