import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from xai_evals.explainer import LIMEExplainer, SHAPExplainer, DlBacktraceTabularExplainer, TorchTabularExplainer, TFTabularExplainer
from sklearn.base import is_regressor, is_classifier
import tensorflow as tf
from tf_explain.core import IntegratedGradients, VanillaGradients, GradCAM, SmoothGrad, OcclusionSensitivity
import numpy as np
import quantus
from tensorflow.keras.utils import to_categorical
from dl_backtrace.tf_backtrace import Backtrace as TFBacktrace
from dl_backtrace.pytorch_backtrace import Backtrace as TorchBacktrace
from xai_evals.utils import backtrace_quantus


class ExplanationMetricsImage:
    """
    A wrapper class to evaluate explanations using Quantus metrics, supporting both PyTorch and TensorFlow models.
    """

    def __init__(self, model, data_loader, framework="torch", device=None, num_classes=10):
        """
        Initializes the wrapper with the model, dataset, and framework.

        Args:
            model (torch.nn.Module or tf.keras.Model): The trained model.
            data_loader (DataLoader or tf.data.Dataset): DataLoader for test data.
            framework (str): Framework of the model ('torch' or 'tensorflow').
            device (torch.device, optional): Device to perform computations (for PyTorch models).
        """
        self.model = model
        self.data_loader = data_loader
        self.framework = framework.lower()
        self.device = device
        self.num_classes = num_classes

        if self.framework == "torch":
            self.model.eval()
            
        # Predefined configuration for metrics
        self.metrics_config = {
            "FaithfulnessCorrelation": quantus.FaithfulnessCorrelation(subset_size=16),
            "MaxSensitivity": quantus.MaxSensitivity(disable_warnings=True),
            "MPRT": quantus.MPRT(disable_warnings=True),
            "SmoothMPRT": quantus.SmoothMPRT(disable_warnings=True),
            "AvgSensitivity": quantus.AvgSensitivity(disable_warnings=True),
            "FaithfulnessEstimate": quantus.FaithfulnessEstimate(perturb_baseline="black", normalise=True),
        }
 
        # Predefined configuration for explanation methods
        if self.framework == "torch":
            self.xai_methods_config = {
                "IntegratedGradients": {"xai_lib": "captum", "method": "IntegratedGradients"},
                "Saliency": {"xai_lib": "captum", "method": "Saliency"},
                "DeepLift": {"xai_lib": "captum", "method": "DeepLift"},
                "GradientShap": {"xai_lib": "captum", "method": "GradientShap"},
                "Occlusion": {"xai_lib": "captum", "method": "Occlusion"},
                'DeepLiftShap': {"xai_lib": "captum", "method": "DeepLiftShap"},
                'InputXGradient': {"xai_lib": "captum", "method": "InputXGradient"},
                'Saliency': {"xai_lib": "captum", "method": "Saliency"},
                'FeatureAblation': {"xai_lib": "captum", "method": "FeatureAblation"},
                'Deconvolution': {"xai_lib": "captum", "method": "Deconvolution"},
                'FeaturePermutation': {"xai_lib": "captum", "method": "FeaturePermutation"},
                'Lime': {"xai_lib": "captum", "method": "Lime"},
                'KernelShap': {"xai_lib": "captum", "method": "KernelShap"},
                'LRP': {"xai_lib": "captum", "method": "LRP"},
                'Gradient': {"xai_lib": "captum", "method": "Gradient"},
                'LayerGradCam': {"xai_lib": "captum", "method": "LayerGradCam"},
                'GuidedGradCam': {"xai_lib": "captum", "method": "GuidedGradCam"},
                'LayerConductance': {"xai_lib": "captum", "method": "LayerConductance"},
                'LayerActivation': {"xai_lib": "captum", "method": "LayerActivation"},
                'InternalInfluence': {"xai_lib": "captum", "method": "InternalInfluence"},
                'LayerGradientXActivation': {"xai_lib": "captum", "method": "LayerGradientXActivation"},
                'Control Var. Sobel Filter': {"xai_lib": "captum", "method": "Control Var. Sobel Filter"},
                'Control Var. Constant': {"xai_lib": "captum", "method": "Control Var. Constant"},
                'Control Var. Random Uniform': {"xai_lib": "captum", "method": "Control Var. Random Uniform"},
            }
        elif self.framework == "tensorflow":
            self.xai_methods_config = {
                "VanillaGradients": {"xai_lib": "tf-explain", "method": "VanillaGradients"},
                "GradCAM": {"xai_lib": "tf-explain", "method": "GradCAM"},
                "GradientsInput": {"xai_lib": "tf-explain", "method": "VanillaGradients"},
                "IntegratedGradients": {"xai_lib": "tf-explain", "method": "IntegratedGradients"},
                "OcclusionSensitivity": {"xai_lib": "tf-explain", "method": "OcclusionSensitivity"},
                "SmoothGrad": {"xai_lib": "tf-explain", "method": "SmoothGrad"},
            }
        elif self.framework == "backtrace":
            print("Backtrace Framework Explaination")
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def evaluate(self, start_idx, end_idx, metric_names, xai_method_name,channel_first=True,softmax=False):
        """
        Evaluates a list of Quantus metrics on a batch of samples.

        Args:
            start_idx (int): Start index of the batch from the data loader.
            end_idx (int): End index of the batch from the data loader.
            metric_names (list): List of metric names to evaluate.
            xai_method_name (str): Name of the XAI method.

        Returns:
            dict: Aggregated scores for each metric.
        """
        # Prepare batch data
        self.softmax = softmax
        self.channel_first = channel_first
        x_batch, y_batch = self._prepare_data(start_idx, end_idx)

        if self.framework == "backtrace":
            if isinstance(self.model, tf.keras.Model):  # TensorFlow model
                self.backtrace_intitial = TFBacktrace(model=self.model)
            elif isinstance(self.model, torch.nn.Module):  # PyTorch model
                self.backtrace_intitial = TorchBacktrace(model=self.model)

        # Wrapper to integrate Backtrace with Quantus
            
            if xai_method_name == "default":
                mode = "default"
                cmode = None
            elif xai_method_name == "contrast-positive":
                mode = "contrast"
                cmode = "Positive"
            elif xai_method_name == "contrast-negative":
                mode = "contrast"
                cmode = "Negative"
            elif xai_method_name == "contrast":
                mode = "contrast"
                cmode = "Positive"  
            else:
                mode = "default"
            
            metrics = {name: self.metrics_config[name] for name in metric_names}
            method_explanation = { 'Backtrace': backtrace_quantus }

            # Compute scores for each metric
            aggregated_scores = {}
            for metric_name, metric in metrics.items():
                raw_scores = metric(
                    model=self.model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    device=self.device,
                    explain_func=backtrace_quantus,
                    explain_func_kwargs={'backtrace': self.backtrace_intitial,'mode':mode, 'cmode':cmode},
                    channel_first=self.channel_first,
                    softmax=self.softmax
                )
                if metric_name == "Continuity":
                    aggregated_scores[metric_name] = self._aggregate_continuity_scores(raw_scores)
                elif metric_name == "MPRT" or metric_name == "SmoothMPRT":
                    aggregated_scores[metric_name] = self._aggregate_MPRT_scores(raw_scores)
                else:
                    aggregated_scores[metric_name] = (
                        np.nanmean(raw_scores) if isinstance(raw_scores, list) else raw_scores
                    )
            return aggregated_scores

        else:
            # Ensure the selected XAI method and metrics exist
            if xai_method_name not in self.xai_methods_config:
                raise ValueError(f"XAI method '{xai_method_name}' is not configured.")
            if not all(metric in self.metrics_config for metric in metric_names):
                raise ValueError("One or more metrics are not configured.")

            # Prepare XAI method and metric instances
            explain_func_kwargs = self.xai_methods_config[xai_method_name]
            metrics = {name: self.metrics_config[name] for name in metric_names}

            # Compute scores for each metric
            aggregated_scores = {}
            for metric_name, metric in metrics.items():
                raw_scores = metric(
                    model=self.model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    device=self.device,
                    explain_func=quantus.explain,
                    explain_func_kwargs=explain_func_kwargs,
                    channel_first=self.channel_first,
                    softmax=self.softmax
                )
                if metric_name == "Continuity":
                    aggregated_scores[metric_name] = self._aggregate_continuity_scores(raw_scores)
                elif metric_name == "MPRT" or metric_name == "SmoothMPRT":
                    aggregated_scores[metric_name] = self._aggregate_MPRT_scores(raw_scores)
                else:
                    aggregated_scores[metric_name] = (
                        np.nanmean(raw_scores) if isinstance(raw_scores, list) else raw_scores
                    )

            return aggregated_scores

    def _prepare_data(self, start_idx, end_idx):
        """
        Prepares a batch of data for Quantus evaluation.

        Args:
            start_idx (int): Start index of the batch.
            end_idx (int): End index of the batch.

        Returns:
            tuple: x_batch (numpy.ndarray, torch.Tensor, or tf.Tensor), y_batch (numpy.ndarray, torch.Tensor, or tf.Tensor)
        """
        x_batch, y_batch = [], []

        # Case 1: PyTorch DataLoader (torch.utils.data.DataLoader)
        if isinstance(self.data_loader, torch.utils.data.DataLoader):
            dataset = self.data_loader.dataset
            for idx in range(start_idx, end_idx):
                image, label = dataset[idx]
                x_batch.append(image.numpy())
                y_batch.append(label)
            
            # Convert to the correct type (either torch.Tensor or numpy array)
            x_batch = np.stack(x_batch)
            y_batch = np.array(y_batch)

        # Case 2: TensorFlow DataLoader (tf.data.Dataset)
        elif isinstance(self.data_loader, tf.data.Dataset):
            #data = list(self.data_loader.unbatch().take(end_idx - start_idx).as_numpy_iterator())
            data = list(self.data_loader.skip(start_idx).unbatch().take(end_idx - start_idx).as_numpy_iterator())
            x_batch, y_batch = zip(*data)
            x_batch = np.stack(x_batch)
            y_batch = np.array(y_batch)

            # Ensure labels are non-categorical integers (0, 1, 2, ...)
            if len(y_batch.shape) > 1 :
                y_batch = to_categorical(y_batch)
                y_batch = np.argmax(y_batch, axis=1)

        # Case 3: Tuple (numpy.ndarray, numpy.ndarray) or (torch.Tensor, torch.Tensor)
        elif isinstance(self.data_loader, tuple):
            data, labels = self.data_loader
            if end_idx > start_idx:
                if start_idx == 0 and end_idx == 1:
                    x_batch = data
                    y_batch = labels
                else:
                    x_batch = data[start_idx:end_idx]
                    y_batch = labels[start_idx:end_idx]

            # Ensure the data and labels are in the correct format
            #if isinstance(x_batch, np.ndarray):
            #x_batch = torch.tensor(x_batch, dtype=torch.float32)

            if isinstance(x_batch, torch.Tensor):
                x_batch = x_batch.numpy()
            elif isinstance(x_batch, tf.Tensor):
                x_batch = x_batch.numpy()
            

            if isinstance(y_batch, torch.Tensor):
                y_batch = y_batch.numpy()
            elif isinstance(y_batch, tf.Tensor):
                y_batch = y_batch.numpy()

            # Ensure labels are non-categorical (integer labels)
            if len(y_batch.shape) > 1 :
                y_batch = to_categorical(y_batch)
                y_batch = np.argmax(y_batch, axis=1)

        # Case 4: Raw PyTorch Dataset (torch.utils.data.Dataset)
        elif isinstance(self.data_loader, torch.utils.data.Dataset):
            x_batch, y_batch = [], []
            for idx in range(start_idx, end_idx):
                image, label = self.data_loader[idx]
                x_batch.append(image)
                y_batch.append(label)

            # Convert lists to tensors or numpy arrays
            x_batch = np.stack(x_batch)
            y_batch = np.array(y_batch)

        # Case 5: Invalid Data Type
        else:
            raise ValueError("Invalid data type. Expected DataLoader, Dataset, or Tuple of arrays.")

        return x_batch, y_batch

    def _aggregate_continuity_scores(self, scores):
        """
        Aggregates the dictionary-based Continuity scores.

        Args:
            scores (dict): Continuity score dictionary.

        Returns:
            float: Aggregated score (mean across all keys and values).
        """
        all_values = []
        for key, value in scores.items():
            if isinstance(value, list):
                all_values.extend(value)
            elif isinstance(value, dict):  # Nested dictionary case
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        all_values.extend(sub_value)
        return np.nanmean(all_values)
    def _aggregate_MPRT_scores(self, scores):
        mprt_dict = {}
        for key, value in scores.items():
            mprt_dict[key] = np.nanmean(np.array(value))
        return mprt_dict

class ExplanationMetricsTabular:
    def __init__(self, 
                 model, 
                 explainer_name, 
                 X_train, 
                 X_test, 
                 y_test, 
                 features, 
                 task, 
                 method="None",
                 metrics=None,
                 scaler=1,
                 thresholding=0.5,
                 start_idx=0, 
                 end_idx=None,
                 subset_samples=False,
                 subset_number=100):
        """
        Initialize ExplanationMetrics class.
        
        - If explainer_name is 'shap' or 'lime', we assume a sklearn-like model.
        - If explainer_name is 'torch', 'tensorflow', or 'backtrace', we assume a deep learning model (TF or Torch).

        For TF/Torch models:
        - If classification, ensure the model outputs probabilities or manually apply softmax/sigmoid before extracting class probabilities.
        - If regression, directly return numeric predictions.
        """
        self.model = model
        self.explainer_name = explainer_name.lower()
        if self.explainer_name=="backtrace":
            self.method = method.lower()
        else:
            self.method = method
        self.X_train = X_train
        if isinstance(X_test, pd.DataFrame):
            self.X_test = X_test.to_numpy()
        elif isinstance(X_test, np.ndarray):
            self.X_test = X_test
        elif isinstance(X_test, torch.Tensor):
            self.X_test = X_test.numpy()
        else:
            raise ValueError("X_test must be a DataFrame or ndarray.")
        
        self.y_test = y_test
        self.features = features
        self.task = task.lower()
        
        self.scaler = scaler
        self.thresholding = thresholding
        self.subset_samples = subset_samples
        self.subset_number = subset_number

        # Set default metrics if none provided
        self.metrics = metrics if metrics else [
            'faithfulness', 'infidelity', 'sensitivity',
            'comprehensiveness', 'sufficiency', 'monotonicity',
            'complexity', 'sparseness'
        ]

        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.X_test)

        # Determine if we're using a sklearn model or TF/PyTorch
        # For SHAP/LIME: sklearn-like model
        # For torch/tensorflow/backtrace: deep learning model
        if self.explainer_name in ['shap', 'lime']:
            self.is_sklearn = True
        else:
            self.is_sklearn = False

        # Initialize explainer
        self.explainer = self._initialize_explainer()

    def _initialize_explainer(self):
        # Determine mode for LIME or others based on the task
        if "regression" in self.task:
            explainer_mode = "regression"
        elif "classification" in self.task:
            explainer_mode = "classification"
        else:
            raise ValueError("Task must be 'binary-classification', 'multiclass-classification', or 'regression'.")

        if self.explainer_name == 'shap':
            return SHAPExplainer(model=self.model, features=pd.Series(self.features), task=self.task, X_train=self.X_train,subset_samples=self.subset_samples,subset_number=self.subset_number)
        elif self.explainer_name == 'lime':
            return LIMEExplainer(model=self.model, features=self.features, task=explainer_mode, X_train=self.X_train)
        elif self.explainer_name == 'torch':
            # Ensure PyTorch model
            if not isinstance(self.model, torch.nn.Module):
                raise ValueError("For 'torch' explainer, model must be a PyTorch model.")
            return TorchTabularExplainer(model=self.model, task=self.task, method=self.method, feature_names=self.features, X_train=self.X_train)
        elif self.explainer_name == 'tensorflow':
            # Ensure TensorFlow model
            if not isinstance(self.model, tf.keras.Model):
                raise ValueError("For 'tensorflow' explainer, model must be a TensorFlow/Keras model.")
            return TFTabularExplainer(model=self.model, task=self.task, method=self.method, feature_names=self.features, X_train=self.X_train)
        elif self.explainer_name == 'backtrace':
            # Backtrace works with TF/PyTorch
            if not (isinstance(self.model, tf.keras.Model) or isinstance(self.model, torch.nn.Module)):
                raise ValueError("For 'backtrace' explainer, model must be a TF/Keras or PyTorch model.")
            return DlBacktraceTabularExplainer(model=self.model, task=self.task, method=self.method, scaler=self.scaler,
                                      thresholding=self.thresholding, feature_names=self.features)
        else:
            raise ValueError("Unsupported explainer name. Choose from 'shap', 'lime', 'torch', 'tensorflow', 'backtrace'.")

    def _predict_proba_sklearn(self, X):
        """
        For sklearn classification models, returns class probabilities.
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not have predict_proba. For classification tasks, this is required.")

    def _predict_sklearn(self, X):
        """
        Predict using sklearn model:
        - Regression: return numeric predictions.
        - Classification: return probability of predicted class.
        """
        if "regression" in self.task:
            return self.model.predict(X).flatten()
        else:
            proba = self._predict_proba_sklearn(X)
            predicted_classes = np.argmax(proba, axis=1)
            return proba[np.arange(len(proba)), predicted_classes]

    def _predict_tf(self, X):
        """
        Predict using a TensorFlow/Keras model.
        
        Assumptions:
        - Regression: model outputs a single numeric value per instance.
        - Classification: 
          - For binary, model might output a single sigmoid probability for the positive class, convert to [1-p, p].
          - For multi-class, model should output probabilities directly. If logits are returned, apply softmax.

        Adjust as needed depending on your model's last layer.
        """
        outputs = self.model.predict(X)
        if "regression" in self.task:
            return outputs.flatten()
        else:
            # Classification
            if outputs.ndim == 1:
                # Binary classification with single output probability
                outputs = outputs.reshape(-1, 1)
                outputs = np.hstack([1 - outputs, outputs])
            
            # If model returns logits, apply softmax:
            # outputs = scipy.special.softmax(outputs, axis=1)  # Uncomment if needed

            predicted_classes = np.argmax(outputs, axis=1)
            return outputs[np.arange(len(outputs)), predicted_classes]

    def _predict_torch(self, X):
        """
        Predict using a PyTorch model.
        
        - Convert X to a torch tensor.
        - Model might return logits or probabilities.
        - If regression: return numeric predictions.
        - If classification:
          - If single output (binary), apply sigmoid and convert to [1-p, p].
          - If multi-class logits, apply softmax to get probabilities.

        Adjust as needed depending on your model's last layer.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(inputs).cpu().numpy()
        
        if "regression" in self.task:
            return outputs.flatten()
        else:
            # Classification
            # If outputs are logits, apply softmax:
            # from scipy.special import softmax
            # outputs = softmax(outputs, axis=1)  # Uncomment if needed

            if outputs.shape[1] == 1:
                # Binary classification: single output probability
                p = outputs.ravel()  # If these are already probabilities
                # If these are logits, apply sigmoid:
                # p = 1/(1+np.exp(-p))
                outputs = np.column_stack([1 - p, p])
            
            predicted_classes = np.argmax(outputs, axis=1)
            return outputs[np.arange(len(outputs)), predicted_classes]

    def _predict_backtrace(self, X):
        """
        Backtrace uses either a TF or PyTorch model. 
        Detect and route to the appropriate prediction function.
        """
        if isinstance(self.model, tf.keras.Model):
            return self._predict_tf(X)
        elif isinstance(self.model, torch.nn.Module):
            return self._predict_torch(X)
        else:
            raise ValueError("Backtrace explainer requires a TF or Torch model.")

    def _predict(self, X):
        """
        Unified predict function that routes to the correct prediction logic.
        """
        if self.is_sklearn:
            return self._predict_sklearn(X)
        else:
            # Torch, TF, or Backtrace
            if self.explainer_name == 'torch':
                return self._predict_torch(X)
            elif self.explainer_name == 'tensorflow':
                return self._predict_tf(X)
            elif self.explainer_name == 'backtrace':
                return self._predict_backtrace(X)

    def _get_explanation(self, start_idx, end_idx):
        attributions_list = []
        if self.explainer_name == 'torch':
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
            for i in range(start_idx, end_idx):
                instance_explanation = self.explainer.explain(X_test_tensor, instance_idx=i)
                attributions_list.append(instance_explanation)
        else:
            for i in range(start_idx, end_idx):
                instance_explanation = self.explainer.explain(self.X_test, instance_idx=i)
                attributions_list.append(instance_explanation)
        return attributions_list

    def _generate_noise(self, shape, epsilon=1e-2):
        return np.random.normal(0, epsilon, size=shape)

    def _infidelity(self, attributions_list, epsilon=1e-2):
        infidelity_scores = []
        for i in range(self.start_idx, self.end_idx):
            x = self.X_test[i:i+1]
            noise = self._generate_noise(x.shape, epsilon=epsilon)[0]
            g_x = attributions_list[i - self.start_idx]['Attribution'].values
            
            f_x = self._predict(x)[0]
            f_x_pert = self._predict((x + noise))[0]

            predicted_impact = np.dot(g_x, noise)
            actual_impact = f_x_pert - f_x

            infidelity_scores.append((predicted_impact - actual_impact)**2)
        return np.mean(infidelity_scores)

    def _sensitivity(self, attributions_list, epsilon=1e-2):
        sensitivity_scores = []
        for i in range(self.start_idx, self.end_idx):
            x = self.X_test[i:i+1]
            noise = self._generate_noise(x.shape, epsilon=epsilon)
            perturbed_x = (x + noise).astype(x.dtype)
            old_attr = attributions_list[i - self.start_idx]['Attribution'].values

            new_explanation = self.explainer.explain(perturbed_x, instance_idx=0)
            new_attr = new_explanation['Attribution'].values

            sensitivity_scores.append(np.linalg.norm(old_attr - new_attr))
        return np.mean(sensitivity_scores)

    def _comprehensiveness(self, attributions_list, k=5):
        comprehensiveness_scores = []
        for i in range(self.start_idx, self.end_idx):
            attrs = attributions_list[i - self.start_idx]['Attribution'].values
            top_k_indices = np.argsort(np.abs(attrs))[-k:]

            x = self.X_test[i].copy()
            x_masked = x.copy()
            x_masked[top_k_indices] = 0

            f_x = self._predict(x.reshape(1, -1))[0]
            f_masked = self._predict(x_masked.reshape(1, -1))[0]
            comprehensiveness_scores.append(f_x - f_masked)
        return np.mean(comprehensiveness_scores)

    def _sufficiency(self, attributions_list, k=5):
        sufficiency_scores = []
        for i in range(self.start_idx, self.end_idx):
            attrs = attributions_list[i - self.start_idx]['Attribution'].values
            top_k_indices = np.argsort(np.abs(attrs))[-k:]

            x = self.X_test[i]
            x_focused = np.zeros_like(x)
            x_focused[top_k_indices] = x[top_k_indices]

            f_x = self._predict(x.reshape(1, -1))[0]
            f_focused = self._predict(x_focused.reshape(1, -1))[0]
            sufficiency_scores.append(f_focused - f_x)
        return np.mean(sufficiency_scores)

    def _monotonicity(self, attributions_list):
        monotonicity_scores = []
        for attributions in attributions_list:
            attrs = attributions['Attribution'].values
            monotonicity_scores.append(np.all(np.diff(attrs) <= 0))
        return np.mean(monotonicity_scores)

    def _complexity(self, attributions_list):
        return np.mean([np.sum(attr['Attribution'].values != 0) for attr in attributions_list])

    def _sparseness(self, attributions_list):
        total_attrs = len(attributions_list) * len(self.features)
        non_zero = np.sum([np.sum(attr['Attribution'].values != 0) for attr in attributions_list])
        return 1 - (non_zero / total_attrs)

    def _faithfulness_correlation(self, attributions_list):
        faithfulness_scores = []
        for i in range(self.start_idx, self.end_idx):
            f_x = self._predict(self.X_test[i:i+1])[0]
            attribution_values = attributions_list[i - self.start_idx]['Attribution'].values

            for j, a_j in enumerate(attribution_values):
                x_pert = self.X_test[i].copy()
                x_pert[j] = 0
                f_pert = self._predict(x_pert.reshape(1, -1))[0]
                change = f_x - f_pert
                faithfulness_scores.append((change, a_j))

        if len(faithfulness_scores) < 2:
            return np.nan

        changes, attrs = zip(*faithfulness_scores)
        corr_matrix = np.corrcoef(changes, attrs)
        return corr_matrix[0, 1]

    def calculate_metrics(self):
        attributions_list = self._get_explanation(self.start_idx, self.end_idx)
        results = {}

        if 'faithfulness' in self.metrics:
            results['faithfulness'] = self._faithfulness_correlation(attributions_list)
            print("Computed faithfulness")
        if 'infidelity' in self.metrics:
            results['infidelity'] = self._infidelity(attributions_list)
            print("Computed infidelity")
        if 'sensitivity' in self.metrics:
            results['sensitivity'] = self._sensitivity(attributions_list)
            print("Computed sensitivity")
        if 'comprehensiveness' in self.metrics:
            results['comprehensiveness'] = self._comprehensiveness(attributions_list)
            print("Computed comprehensiveness")
        if 'sufficiency' in self.metrics:
            results['sufficiency'] = self._sufficiency(attributions_list)
            print("Computed sufficiency")
        if 'monotonicity' in self.metrics:
            results['monotonicity'] = self._monotonicity(attributions_list)
            print("Computed monotonicity")
        if 'complexity' in self.metrics:
            results['complexity'] = self._complexity(attributions_list)
            print("Computed complexity")
        if 'sparseness' in self.metrics:
            results['sparseness'] = self._sparseness(attributions_list)
            print("Computed sparseness")

        return pd.DataFrame(results, index=[0])
