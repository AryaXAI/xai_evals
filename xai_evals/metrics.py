import pandas as pd
import numpy as np
from xai_evals.explainer import LIMEExplainer, SHAPExplainer

class ExplanationMetrics:
    def __init__(self, 
                 model, 
                 explainer_name, 
                 X_train, 
                 X_test, 
                 y_test, 
                 features, 
                 task, 
                 metrics=None, 
                 start_idx=0, 
                 end_idx=None):
        """
        Initialize the ExplanationMetrics class.

        :param model: Trained model (e.g., XGBoost, RandomForest, etc.)
        :param explainer_name: Name of the explanation method ('shap' or 'lime')
        :param X_train: Training data (used for initializing explainers like SHAP)
        :param X_test: Test data (numpy array or pandas DataFrame)
        :param y_test: True labels or targets of test data
        :param features: List of feature names
        :param task: Task type ('binary', 'multiclass', or 'regression')
        :param metrics: List of metrics to calculate. Defaults to a set of common ones.
        :param start_idx: Starting index of the dataset to evaluate
        :param end_idx: Ending index (if None, goes until the end of X_test)
        """
        self.model = model
        self.explainer_name = explainer_name
        self.X_train = X_train

        if isinstance(X_test, pd.DataFrame):
            self.X_test = X_test.to_numpy()
        elif isinstance(X_test, np.ndarray):
            self.X_test = X_test
        else:
            raise ValueError("X_test must be a DataFrame or ndarray")
        
        self.y_test = y_test
        self.features = features
        self.task = task

        # Default metrics if none provided
        self.metrics = metrics if metrics else [
            'faithfulness', 'infidelity', 'sensitivity',
            'comprehensiveness', 'sufficiency', 'monotonicity',
            'complexity', 'sparseness'
        ]

        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.X_test)

        # Initialize explainer
        self.explainer = self._initialize_explainer()

    def _initialize_explainer(self):
        """
        Initialize the explainer based on the explainer_name ('shap' or 'lime').
        """
        if self.explainer_name == 'shap':
            return SHAPExplainer(model=self.model, features=self.features, task=self.task, X_train=self.X_train)
        elif self.explainer_name == 'lime':
            if self.task == 'binary':
                lime_task = 'classification'
            else:
                lime_task = self.task
            return LIMEExplainer(model=self.model, features=self.features, task=lime_task, X_train=self.X_train)
        else:
            raise ValueError("Unsupported explainer name. Choose 'shap' or 'lime'.")

    def _predict_proba(self, X):
        """
        For classification tasks, returns the probability for each class.
        For regression tasks, this function should not be called.
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("predict_proba is not available for this model. Are you handling regression incorrectly?")

    def _predict(self, X):
        """
        For regression tasks, returns the predicted value.
        For classification tasks, returns the predicted class probabilities.

        To handle tasks consistently:
        - If regression: returns a scalar prediction per instance.
        - If classification: returns the predicted probability of the chosen (predicted) class.
        """
        if self.task == 'regression':
            # Regression returns a single value prediction
            return self.model.predict(X).flatten()
        else:
            # Classification
            # We choose the predicted class per instance dynamically
            proba = self._predict_proba(X)
            predicted_classes = np.argmax(proba, axis=1)
            # Extract the probability of the predicted class for each instance
            return proba[np.arange(len(proba)), predicted_classes]

    def _get_explanation(self, start_idx, end_idx):
        """
        Generate explanations for a range of instances.
        Returns a list of DataFrames (one per instance).
        """
        attributions_list = []
        for i in range(start_idx, end_idx):
            instance_explanation = self.explainer.explain(self.X_test, instance_idx=i)
            attributions_list.append(instance_explanation)
        return attributions_list

    def _generate_perturbations(self, X, epsilon=1e-2):
        """
        Generate perturbations by adding Gaussian noise.
        """
        return X + np.random.normal(0, epsilon, size=X.shape)

    def _infidelity(self, attributions_list):
        """
        Infidelity:
        (f(x+ε)-f(x) - g(x)·ε)² averaged over perturbations and instances.
        """
        infidelity_scores = []
        perturbations = self._generate_perturbations(self.X_test)
        for i in range(self.start_idx, self.end_idx):
            perturbation = perturbations[i]
            predicted_impact = np.dot(perturbation, attributions_list[i]['Attribution'].values)

            f_x = self._predict(self.X_test[i:i+1])[0]
            f_x_pert = self._predict((self.X_test[i] + perturbation).reshape(1, -1))[0]
            actual_impact = f_x_pert - f_x

            infidelity_scores.append((predicted_impact - actual_impact)**2)
        return np.mean(infidelity_scores)

    def _sensitivity(self, attributions_list, epsilon=1e-2):
        """
        Sensitivity:
        E[||g(x)-g(x+ε)||] over perturbations and instances.
        """
        sensitivity_scores = []
        for i in range(self.start_idx, self.end_idx):
            perturbed_instance = self.X_test[i] + np.random.normal(0, epsilon, size=self.X_test[i].shape)
            # Recompute explanation on perturbed instance
            new_explanation = self.explainer.explain(perturbed_instance.reshape(1, -1), instance_idx=0)

            old_attr = attributions_list[i]['Attribution'].values
            new_attr = new_explanation['Attribution'].values
            sensitivity_scores.append(np.linalg.norm(old_attr - new_attr))
        return np.mean(sensitivity_scores)

    def _comprehensiveness(self, attributions_list, k=5):
        """
        Comprehensiveness:
        f(x) - f(x with top-k features removed)
        """
        comprehensiveness_scores = []
        for i in range(self.start_idx, self.end_idx):
            attributions = attributions_list[i]['Attribution'].values
            top_k_indices = np.argsort(attributions)[-k:]

            masked_instance = self.X_test[i].copy()
            masked_instance[top_k_indices] = 0

            f_x = self._predict(self.X_test[i:i+1])[0]
            f_masked = self._predict(masked_instance.reshape(1, -1))[0]
            comprehensiveness_scores.append(f_x - f_masked)
        return np.mean(comprehensiveness_scores)

    def _sufficiency(self, attributions_list, k=5):
        """
        Sufficiency:
        f(x with only top-k) - f(x)
        """
        sufficiency_scores = []
        for i in range(self.start_idx, self.end_idx):
            attributions = attributions_list[i]['Attribution'].values
            top_k_indices = np.argsort(attributions)[-k:]

            focused_instance = np.zeros_like(self.X_test[i])
            focused_instance[top_k_indices] = self.X_test[i][top_k_indices]

            f_x = self._predict(self.X_test[i:i+1])[0]
            f_focused = self._predict(focused_instance.reshape(1, -1))[0]
            sufficiency_scores.append(f_focused - f_x)
        return np.mean(sufficiency_scores)

    def _monotonicity(self, attributions_list):
        """
        Monotonicity:
        Check if attributions are non-increasing as listed.
        a_1 >= a_2 >= ... >= a_n
        """
        monotonicity_scores = []
        for attributions in attributions_list:
            attrs = attributions['Attribution'].values
            # Check non-increasing sequence
            monotonicity_scores.append(np.all(np.diff(attrs) <= 0))
        return np.mean(monotonicity_scores)

    def _complexity(self, attributions_list):
        """
        Complexity:
        Average number of non-zero attributions.
        """
        return np.mean([np.sum(attr['Attribution'].values != 0) for attr in attributions_list])

    def _sparseness(self, attributions_list):
        """
        Sparseness:
        Fraction of attributions that are zero.
        sparseness = 1 - (non_zero_attributions / total_attributions)
        """
        total_attrs = len(attributions_list) * len(self.features)
        non_zero = np.sum([np.sum(attr['Attribution'].values != 0) for attr in attributions_list])
        return 1 - (non_zero / total_attrs)

    def _faithfulness_correlation(self, attributions_list):
        """
        Faithfulness:
        Correlation between attribution values and actual changes in prediction when removing each feature.

        For each feature j:
        change_j = f(x) - f(x_{-j})
        a_j = attribution for feature j
        Then compute Pearson's correlation over all features and instances.
        """
        faithfulness_scores = []
        for i in range(self.start_idx, self.end_idx):
            f_x = self._predict(self.X_test[i:i+1])[0]
            attribution_values = attributions_list[i]['Attribution'].values

            for j, attribution in enumerate(attribution_values):
                perturbed_instance = self.X_test[i].copy()
                perturbed_instance[j] = 0
                f_pert = self._predict(perturbed_instance.reshape(1, -1))[0]
                change = f_x - f_pert
                faithfulness_scores.append((change, attribution))

        if len(faithfulness_scores) < 2:
            # Not enough data to compute correlation
            return np.nan

        changes, attrs = zip(*faithfulness_scores)
        return np.corrcoef(changes, attrs)[0, 1]

    def calculate_metrics(self):
        """
        Calculate the specified metrics and return as a DataFrame.
        """
        attributions_list = self._get_explanation(self.start_idx, self.end_idx)
        results = {}

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
        if 'complexity' in self.metrics:
            results['complexity'] = self._complexity(attributions_list)
        if 'sparseness' in self.metrics:
            results['sparseness'] = self._sparseness(attributions_list)

        return pd.DataFrame(results, index=[0])