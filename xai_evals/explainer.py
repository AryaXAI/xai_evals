import shap
import lime.lime_tabular
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV, RidgeClassifier, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import tensorflow as tf
import xgboost as xgb
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from dl_backtrace.tf_backtrace import Backtrace as TFBacktrace
from dl_backtrace.pytorch_backtrace import Backtrace as TorchBacktrace
from captum.attr import (
    IntegratedGradients, Saliency, DeepLift, GuidedBackprop, InputXGradient,
    DeepLiftShap, GradientShap, Deconvolution, GuidedGradCam, LayerGradCam,
    Occlusion, FeatureAblation, ShapleyValueSampling, ShapleyValues, NoiseTunnel
)
from captum.attr import LayerAttribution

class DlBacktraceImageExplainer:
    def __init__(self, model,):
        self.model = model
        # Automatically detect whether the model is TensorFlow or PyTorch
        if isinstance(self.model, tf.keras.Model):  # TensorFlow model
            self.framework = "tensorflow"
            self.backtrace = TFBacktrace(model=model)
            self.inp_layer = self.backtrace.model_resource["inputs"][0]
        elif isinstance(self.model, torch.nn.Module):  # PyTorch model
            self.framework = "pytorch"
            self.backtrace = TorchBacktrace(model=model)
            self.inp_layer = 'identity'
        else:
            raise ValueError("Unsupported model type. Only TensorFlow and PyTorch models are supported.")
    
    def get_last_conv_layer(self,model):
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv
    
    def explain(self, test_data, instance_idx=0,mode='default',scaler=1,thresholding=0,task='binary-classification',contrast_mode='Positive'):
        relevance = None
        self.test_data = test_data
        self.instance_idx = instance_idx
        self.mode = mode
        self.scaler = scaler
        self.thresholding = thresholding
        self.task = task
        if self.mode == 'contrast':
            self.contrast_mode = contrast_mode
        else:
            self.contrast_mode = None

        if isinstance(self.test_data, torch.utils.data.DataLoader):
            dataset = self.test_data.dataset
            image, label = dataset[self.instance_idx]
            instance = image.unsqueeze(0).to(next(self.model.parameters()).device)  # [1, C, H, W]
            label = label

        # If input is a Tensor or NumPy array (single image), use the provided data
        elif isinstance(self.test_data, torch.Tensor):
            image = self.test_data[self.instance_idx] if instance_idx is not None else test_data
            instance = image.unsqueeze(0).to(next(self.model.parameters()).device)  # [1, C, H, W]
            label = None  # For single image, label can be None or passed externally

        elif isinstance(self.test_data, np.ndarray):
            image = self.test_data[self.instance_idx] if instance_idx is not None else test_data
            instance = np.expand_dims(image, 0)
            label = None  # For single image, label can be None or passed externally

        else:
            raise ValueError("Invalid test data type. Must be Tensor, NumPy array, or DataLoader.")

        if self.framework == "pytorch":
            instance = torch.tensor(instance, dtype=torch.float32)

        # Step 1: Get the layer outputs using the Backtrace model
        layer_outputs = self.backtrace.predict(instance)
        
        # Step 2: Get the layer-wise relevance using DLBacktrace
        relevance = self.backtrace.eval(
            layer_outputs,
            mode=self.mode,
            scaler=self.scaler,
            thresholding=self.thresholding,
            task=self.task
        )

        if self.framework == "pytorch":
            if self.mode == 'default':
                attr_2d = np.mean(relevance[self.inp_layer],axis=-3)
            elif self.mode == 'contrast':
                attr_2d = np.mean(relevance[self.inp_layer][self.contrast_mode],axis=-3)
            else:
                attr_2d = np.mean(relevance[self.inp_layer],axis=-3)
        else:
            # Aggregate attributions over channels (RGB) by taking the mean
            if self.mode == 'default':
                attr_2d = np.mean(relevance[self.inp_layer],axis=-1)
            elif self.mode == 'contrast':
                attr_2d = np.mean(relevance[self.inp_layer][self.contrast_mode],axis=-1)
            else:
                attr_2d = np.mean(relevance[self.inp_layer],axis=-1)

        return attr_2d

class TorchImageExplainer:
    """
    Wrapper for Captum explainers.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.last_conv_layer = self.get_last_conv_layer(model)

    def get_last_conv_layer(self,model):
        """
        Retrieves the last convolutional layer of the model.
        
        Args:
            model (nn.Module): The CNN model.
        
        Returns:
            nn.Module: The last convolutional layer.
        """
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def explain(self, testdata, idx=None, method="grad_cam", task="classification"):
        """
        Generates attributions for a specific test sample using a specified method.
        
        Args:
            testloader (DataLoader): DataLoader for test data.
            idx (int): Index of the test sample.
            method (str): Attribution method to use.
            task (str): Task type, default is "classification".
        
        Returns:
            np.ndarray: 2D attribution map.
        """
        # If input is a DataLoader (batch of images), extract the specific image and label
        if isinstance(testdata, torch.utils.data.DataLoader):
            dataset = testdata.dataset
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(next(self.model.parameters()).device)  # [1, C, H, W]
            label = label

        # If input is a Tensor or NumPy array (single image), use the provided data
        elif isinstance(testdata, torch.Tensor):
            image = testdata[idx] if idx is not None else testdata
            image = image.unsqueeze(0).to(next(self.model.parameters()).device)  # [1, C, H, W]
            label = None  # For single image, label can be None or passed externally

        elif isinstance(testdata, np.ndarray):
            image = testdata[idx] if idx is not None else testdata
            image = torch.tensor(image).float().unsqueeze(0).to(next(self.model.parameters()).device)  # [1, C, H, W]
            label = None  # For single image, label can be None or passed externally

        else:
            raise ValueError("Invalid test data type. Must be Tensor, NumPy array, or DataLoader.")


        with torch.no_grad():
            logits = self.model(image)
            preds = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(preds, dim=1).item()

        baseline = torch.zeros_like(image).to(image.device)

        if method == "integrated_gradients":
            explainer = IntegratedGradients(self.model)
            attributions = explainer.attribute(image, baselines=baseline, target=pred_class)
        elif method == "saliency":
            explainer = Saliency(self.model)
            attributions = explainer.attribute(image, target=pred_class)
        elif method == "deep_lift":
            explainer = DeepLift(self.model)
            attributions = explainer.attribute(image, baselines=baseline, target=pred_class)
        elif method == "guided_backprop":
            explainer = GuidedBackprop(self.model)
            attributions = explainer.attribute(image, target=pred_class)
        elif method == "input_x_gradient":
            explainer = InputXGradient(self.model)
            attributions = explainer.attribute(image, target=pred_class)
        elif method == "deep_lift_shap":
            explainer = DeepLiftShap(self.model)
            #baselines = [torch.randn_like(image)*0.1 for _ in range(5)]
            baseline = torch.randn_like(image).repeat(5, 1, 1, 1) * 0.1  # Shape: [5, C, H, W]
            attributions = explainer.attribute(image, baselines=baseline, target=pred_class)
        elif method == "gradient_shap":
            explainer = GradientShap(self.model)
            #baselines = [torch.randn_like(image)*0.1 for _ in range(5)]
            baseline = torch.randn_like(image) * 0.1
            attributions = explainer.attribute(image, baselines=baseline, target=pred_class, stdevs=0.09)
        elif method == "deconvolution":
            explainer = Deconvolution(self.model)
            attributions = explainer.attribute(image, target=pred_class)
        elif method == "guided_gradcam":
            explainer = GuidedGradCam(self.model, self.last_conv_layer)
            attributions = explainer.attribute(image, target=pred_class)
        elif method == "layer_gradcam":
            explainer = LayerGradCam(self.model, self.last_conv_layer)
            attributions = explainer.attribute(image, target=pred_class)
            attributions = LayerAttribution.interpolate(attributions, image.shape[2:])
        elif method == "occlusion":
            explainer = Occlusion(self.model)
            attributions = explainer.attribute(
                image, target=pred_class, baselines=baseline, 
                sliding_window_shapes=(3,8,8)
            )
        elif method == "feature_ablation":
            explainer = FeatureAblation(self.model)
            attributions = explainer.attribute(image, target=pred_class, baselines=baseline)
        elif method == "shapley_value_sampling":
            explainer = ShapleyValueSampling(self.model)
            attributions = explainer.attribute(image, target=pred_class, baselines=baseline, n_samples=50)
        elif method == "shapley_values":
            explainer = ShapleyValues(self.model)
            attributions = explainer.attribute(image, target=pred_class, baselines=baseline)
        elif method == "noise_tunnel":
            base_explainer = IntegratedGradients(self.model)
            explainer = NoiseTunnel(base_explainer)
            attributions = explainer.attribute(
                image, 
                target=pred_class, 
                nt_type='smoothgrad',   # Type of noise (can also be 'vanilla')
                stdevs=0.02            # Standard deviation of noise
            )
        else:
            raise ValueError("Unsupported method")

        # Aggregate attributions over channels (RGB) by taking the mean
        attr_2d = attributions.mean(dim=1).squeeze(0).detach().cpu().numpy()
        return attr_2d

class TFImageExplainer:
    """
    Wrapper for tf-explain explainers for TensorFlow models.
    Handles both single image and dataset cases.
    """
    def __init__(self, model):
        """
        Initializes the explainer with a TensorFlow model.
        """
        self.model = model
        self.model.trainable = False  # Set model to inference mode
        self.last_conv_layer = self._get_last_conv_layer()
        self.is_single_image = False

    def _get_last_conv_layer(self):
        """
        Get the last convolutional layer of the model for GradCAM.
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        raise ValueError("No convolutional layer found in the model")

    def explain(self, testset, idx=None, method="grad_cam",task=None,label=0):
        """
        Generates attributions for a specific test sample using a specified method.
        """
        # Check if the input is a single image (NumPy array or TensorFlow tensor)
                # Determine class index
        if isinstance(testset, (np.ndarray, tf.Tensor)):
            # If idx is None, process the entire single image
            if idx is not None:
                raise ValueError("For single image input, idx should be None.")
            image = testset
            self.is_single_image = True
            class_index = int(label)
        elif isinstance(testset, tf.data.Dataset):
            # If it's a TensorFlow Dataset, handle idx
            self.is_single_image = False
            batch = list(testset.take(idx // testset.cardinality().numpy() + 1))[-1]  # Retrieve the required batch
            images, labels = batch

            # Get the specific image and label from the batch
            image = images[idx % images.shape[0]]
            label = labels[idx % images.shape[0]]
            class_index = int(label)
        else:
            raise ValueError("Unsupported input type. Expected a NumPy array, TensorFlow tensor, or Dataset.")

        image = tf.expand_dims(image, axis=0)  # This ensures the image has shape (1, height, width, channels)
        image_numpy = image.numpy()  # Convert to NumPy for compatibility with tf-explain

        # Choose the appropriate explanation method
        if method == "grad_cam":
            explainer = GradCAM()
            if class_index is None:
                raise ValueError("class_index must be provided for GradCAM in classification tasks.")
            attributions = explainer.explain((image_numpy, None), self.model, class_index=class_index, layer_name=self.last_conv_layer.name)

        elif method == "occlusion":
            explainer = OcclusionSensitivity()
            if class_index is None:
                raise ValueError("class_index must be provided for OcclusionSensitivity in classification tasks.")
            attributions = explainer.explain((image_numpy, None), self.model, class_index=class_index, patch_size=8)

        else:
            raise ValueError(f"Unsupported method: {method}")

        # Convert the attribution to 2D by averaging over channels (RGB)
        attr_2d = np.mean(attributions, axis=-1)  # This collapses the channels to provide a 2D map
        return attr_2d

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
        elif isinstance(self.model, torch.nn.Module): 
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
        if self.task == "binary-classification" or "binary" in self.task:
            if len(attributions.shape) == 3:
                idx = np.argmax(self._model_predict(x_instance))
                attributions = attributions[:,:,idx]
        elif self.task == "multiclass-classification" or "multiclass" in self.task:
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
            if "<=" in feature or "<" in feature or ">" in feature or "=" in feature:
                if not "<=" in feature and "=" in feature:
                    feature = feature.split("=")
                    original_feature = next(word.strip() for word in feature if word.strip() in self.features)
                else:
                    original_feature = next(word.strip() for word in feature.split() if word.strip() in self.features)
            else:
                original_feature = feature
            feature_value = x_instance[original_feature].values[0]
            original_attributions.append((original_feature, feature_value, attribution))
        attribution_df = pd.DataFrame(original_attributions, columns=['Feature', 'Value', 'Attribution'])
        attribution_df['Attribution'] = attribution_df['Attribution'].abs()
        return attribution_df

class TFTabularExplainer:
    def __init__(self, model, method, feature_names, X_train=None, task="binary-classification"):
        """
        Initialize a TF explainer for Integrated Gradients, SHAP, and LIME.
        :param model: Trained TensorFlow/Keras model.
        :param method: 'integrated_gradients_tf', 'shap_kernel', 'shap_deep', or 'lime'.
        :param feature_names: List of feature names.
        :param X_train: Training data (required for SHAP KernelExplainer and LIME).
        :param task: "classification" or "regression".
        """
        self.model = model
        self.feature_names = feature_names
        self.method = method.lower()
        self.X_train = X_train
        self.task = task

        # Validate method
        if self.method not in ["shap_kernel", "shap_deep", "lime"]:
            raise ValueError("Only 'shap_kernel', 'shap_deep', and 'lime' are supported.")

        if self.method == "shap_kernel" and self.X_train is None:
            raise ValueError("X_train is required for SHAP KernelExplainer.")
        if self.method == "lime" and self.X_train is None:
            raise ValueError("X_train is required for LIME.")

        # Initialize explainers as needed
        if self.method == "shap_kernel":
            self.explainer = shap.KernelExplainer(self._model_predict, self.X_train)
        elif self.method == "shap_deep":
            #background = self.X_train if isinstance(self.X_train, tf.Tensor) else tf.convert_to_tensor(self.X_train, tf.float32)
            self.explainer = shap.DeepExplainer(self.model, self.X_train)
        elif self.method == "lime":
            X_train_np = self.X_train if isinstance(self.X_train, np.ndarray) else self.X_train.numpy()
            if self.task == "Regression" or self.task == "regression":
                self.shap_task = "regression"
            else:
                self.shap_task = "classification"

            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_np,
                feature_names=self.feature_names,
                class_names=None if self.task == "regression" else [0, 1],
                mode=self.shap_task
            )

    def _model_predict(self, inputs):
        """
        Model prediction function for SHAP and LIME.
        Ensures outputs are in the expected format (e.g., probabilities for classification).
        """
        inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
        outputs = self.model.predict(inputs_tf)  # shape: [batch, output_dim]

        if self.task == "classification" or self.task == "binary-classification" or self.task=="multiclass-classification":
            if outputs.shape[1] == 1:
            # Convert single-probability output to two-class probabilities
                return np.hstack([1 - outputs, outputs])
        return outputs

    def explain(self, inputs, target=0, instance_idx=None):
        """
        Explain the predictions for given inputs.
        
        Args:
            inputs: A single instance or a batch of instances. If a batch is provided, specify instance_idx.
            baseline: Baseline for integrated gradients if needed.
            target: Target class index for integrated gradients or multi-class SHAP.
            instance_idx: Index of the instance to explain within the batch.
        
        Returns:
            A DataFrame with feature names, input values, and attributions.
        """
        if isinstance(inputs, tf.Tensor):
            inputs_np = inputs.numpy()
            print("1")
        else:
            inputs_np = np.array(inputs, dtype=np.float32)

        if instance_idx is None:
            instance_idx = 0

        single_input = inputs_np[instance_idx:instance_idx + 1]  # shape: [1, features]


        if self.method == "shap_kernel" or self.method == "shap_deep":
            shap_values = self.explainer.shap_values(single_input)
            attributions = shap_values
            #print(self.task,self.shap_type,attributions.shape)
            if self.task == "binary-classification" or "binary" in self.task:
                if len(attributions.shape) == 3:
                    idx = np.argmax(self._model_predict(single_input))
                    if self.method == "shap_deep":
                        attributions = attributions[:,:,0]
                    else:
                        attributions = attributions[:,:,idx]
            elif self.task == "multiclass-classification" or "multiclass" in self.task:
                if len(attributions.shape) == 3:
                    idx = np.argmax(self._model_predict(single_input))
                    attributions = attributions[:,:,idx]
            elif self.task == "multi-label-classification":
                pass
            else:
                pass
        elif self.method == "lime":
            explanation = self.explainer.explain_instance(single_input[0], self._model_predict)
            return self._map_binned_to_original(explanation.as_list(), single_input[0])

        # Ensure attributions is at least 2D: [1, features]
        attributions = np.array(attributions)
        if attributions.ndim == 1:
            attributions = attributions[np.newaxis, :]

        print(single_input.shape,attributions.shape,self.feature_names)
        # Create a results DataFrame
        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Value": single_input[0],
            "Attribution": attributions[0]
        })
        df = df.sort_values(by="Attribution", key=np.abs, ascending=False)
        return df

    def _map_binned_to_original(self, attributions, x_instance):
        """
        Map the binned features from LIME to original features.
        """
        original_attributions = []
        
        for feature, attribution in attributions:
            if "<=" in feature or "<" in feature or ">" in feature:
                original_feature = next(
                    (word.strip() for word in feature.split() if word.strip() in self.feature_names),
                    feature
                )
            else:
                original_feature = feature

            feature_index = self.feature_names.index(original_feature)
            feature_value = x_instance[feature_index]

            original_attributions.append((original_feature, feature_value, attribution))

        attribution_df = pd.DataFrame(
            original_attributions,
            columns=["Feature", "Value", "Attribution"]
        )
        attribution_df = attribution_df.sort_values(by="Attribution", key=np.abs, ascending=False)
        return attribution_df

class TorchTabularExplainer:
    def __init__(self, model, method, feature_names, X_train=None, task="binary-classification"):
        """
        Initialize the unified explainer for Captum, LIME, and SHAP.
        :param model: Trained PyTorch model (for Captum).
        :param method: Explanation method. Options:
            - Captum: 'integrated_gradients', 'deep_lift', 'gradient_shap',
                      'saliency', 'input_x_gradient', 'guided_backprop'
            - SHAP: 'shap_kernel', 'shap_deep'
            - LIME: 'lime'
        :param feature_names: List of feature names corresponding to input features.
        :param X_train: Training data (required for SHAP KernelExplainer and LIME).
        :param task: Task type ('classification' or 'regression').
        """
        self.model = model
        self.feature_names = feature_names
        self.method = method.lower()
        self.X_train = X_train
        self.task = task

        # Select the attribution method
        self.explainer = self._select_method()

    def _select_method(self):
        """
        Select the explanation method based on the specified input.
        """
        if self.method in ["integrated_gradients", "deep_lift", "gradient_shap", "saliency", "input_x_gradient", "guided_backprop"]:
            return self._get_captum_method()
        elif self.method == "shap_kernel":
            if self.X_train is None:
                raise ValueError("X_train is required for SHAP KernelExplainer.")
            return shap.KernelExplainer(self._model_predict, self.X_train)
        elif self.method == "shap_deep":
            if isinstance(self.X_train, torch.Tensor):
                pass
            else:
                self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
            return shap.DeepExplainer(self.model, self.X_train)
        elif self.method == "lime":
            if self.X_train is None:
                raise ValueError("X_train is required for LIME.")
            return lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train,
                feature_names=self.feature_names,
                class_names=None if self.task == "regression" else [0, 1],
                mode=self.task
            )
        else:
            raise ValueError(f"Unsupported explanation method: {self.method}")

    def _get_captum_method(self):
        """
        Initialize the Captum method.
        """
        if self.method == "integrated_gradients":
            return IntegratedGradients(self.model)
        elif self.method == "deep_lift":
            return DeepLift(self.model)
        elif self.method == "gradient_shap":
            return GradientShap(self.model)
        elif self.method == "saliency":
            return Saliency(self.model)
        elif self.method == "input_x_gradient":
            return InputXGradient(self.model)
        elif self.method == "guided_backprop":
            return GuidedBackprop(self.model)

    def _model_predict(self, inputs):
        """
        Wrapper for model prediction to ensure compatibility with SHAP and LIME.
        Converts PyTorch tensors to numpy arrays for compatibility.
        Ensures output includes probabilities for both classes in binary classification.
        """
        self.model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32)
            outputs = self.model(inputs).numpy()  # Get model outputs as numpy array

        if self.task == "classification":
            # Ensure binary classification outputs both probabilities: [P(y=0), P(y=1)]
            return np.hstack([1 - outputs, outputs])  # Convert single probability to two-class format
        return outputs.squeeze()

    def explain(self, inputs, baseline=None, target=0, instance_idx=None):
        """
        Explain the predictions for the given inputs.
        :param inputs: Input tensor or array for Captum/SHAP/LIME.
        :param baseline: Baseline tensor for Captum methods requiring baselines.
        :param target: Target class index for Captum methods.
        :param instance_idx: Index of the instance to explain (for LIME and SHAP).
        :return: DataFrame with feature names, input values, and attributions.
        """
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.detach().numpy()  # Convert PyTorch tensor to NumPy array
        else:
            inputs_np = inputs

        if self.method in ["integrated_gradients", "gradient_shap"]:
            if instance_idx is not None:
                inputs = inputs[instance_idx:instance_idx + 1]

            if baseline is None:
                if isinstance(inputs, torch.Tensor):
                    pass
                else:
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                baseline = torch.zeros_like(inputs)  # Default baseline
            attributions = self.explainer.attribute(inputs, baselines=baseline, target=target)
        elif self.method in ["deep_lift", "saliency", "input_x_gradient", "guided_backprop"]:
            if instance_idx is not None:
                inputs = inputs[instance_idx:instance_idx + 1]
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs)
            attributions = self.explainer.attribute(inputs, target=target)
        elif self.method == "shap_kernel":
            if instance_idx is not None:
                inputs_np = inputs_np[instance_idx:instance_idx + 1]
            shap_values = self.explainer.shap_values(inputs_np)
            attributions = shap_values[0]  # For binary classification, use first class
        elif self.method == "shap_deep":
            if instance_idx is not None:
                inputs = inputs[instance_idx:instance_idx + 1]
            shap_values = self.explainer.shap_values(inputs)
            attributions = shap_values[0]  # For binary classification
        elif self.method == "lime":
            if instance_idx is None :
                instance_idx = 0
            explanation = self.explainer.explain_instance(
                inputs_np[instance_idx], self._model_predict
            )
            attributions = explanation.as_list()
            return self._map_binned_to_original(attributions, inputs_np[instance_idx])

        # Convert attributions to a DataFrame
        attributions = attributions.detach().numpy() if isinstance(attributions, torch.Tensor) else attributions
        
        if instance_idx is None :
            instance_idx = 0

        results = pd.DataFrame({
            "Feature": self.feature_names,
            "Value": inputs_np[instance_idx],
            "Attribution": attributions[0] if len(attributions.shape) > 1 else attributions
        }).sort_values(by="Attribution", key=abs, ascending=False)

        return results

    def _map_binned_to_original(self, attributions, x_instance):
        """
        Map the binned features to the original features and return a DataFrame.
        """
        original_attributions = []
        
        for feature, attribution in attributions:
            # Extract the original feature name by removing binning conditions
            if "<=" in feature or "<" in feature or ">" in feature:
                original_feature = next(
                    (word.strip() for word in feature.split() if word.strip() in self.feature_names),
                    feature  # Fall back to the full feature name if no match
                )
            else:
                original_feature = feature

            # Get the corresponding feature value from the instance
            feature_index = self.feature_names.index(original_feature)
            feature_value = x_instance[feature_index]

            # Append the original feature name, feature value, and attribution
            original_attributions.append((original_feature, feature_value, attribution))
        
        # Convert to a DataFrame for better readability
        attribution_df = pd.DataFrame(
            original_attributions,
            columns=["Feature", "Value", "Attribution"]
        )
        attribution_df = attribution_df.sort_values(by="Attribution", key=abs, ascending=False)
        return attribution_df

class DlBacktraceTabularExplainer:
    def __init__(self, model, feature_names, task="binary-classification", scaler=1, thresholding=0.5, method="default", X_train=None):
        """
        Initialize the BacktraceExplainer with model, framework detection, task type, and other evaluation parameters.
        :param model: Trained model (TensorFlow/Keras or PyTorch model)
        :param task: Type of task - "binary-classification", "multi-class classification", "segmentation", etc.
        :param method: The mode of evaluation - "default" or "contrastive"
        :param scaler: Total / Starting Relevance at the Last Layer (Integer, Default: 1)
        :param thresholding: Threshold for segmentation tasks (Default: 0.5)
        :param X_train: Training data used for the model (for some explainers)
        """
        self.model = model
        self.task = task
        self.mode = method
        self.scaler = scaler
        self.thresholding = thresholding
        self.feature_names = feature_names

        if self.mode == "default":
            self.mode = "default"
            self.cmode = None
        elif self.mode == "contrast-positive":
            self.mode = "contrast"
            self.cmode = "Positive"
        elif self.mode == "contrast-negative":
            self.mode = "contrast"
            self.cmode = "Negative"
        elif self.mode == "contrast":
            self.mode = "contrast"
            self.cmode = "Positive"  
        else:
            self.mode = "default"

        # Automatically detect whether the model is TensorFlow or PyTorch
        if isinstance(self.model, tf.keras.Model):  # TensorFlow model
            self.framework = "tensorflow"
            self.backtrace = TFBacktrace(model=model)
            self.inp_layer = self.backtrace.model_resource["inputs"][0]
        elif isinstance(self.model, torch.nn.Module):  # PyTorch model
            self.framework = "pytorch"
            self.backtrace = TorchBacktrace(model=model)
            self.inp_layer = 'identity'
        else:
            raise ValueError("Unsupported model type. Only TensorFlow and PyTorch models are supported.")

    def _prepare_data(self, test_data):
        """
        Ensure the input data is in the correct format for evaluation.
        """
        if isinstance(test_data, np.ndarray):
            return test_data
        elif isinstance(test_data, pd.DataFrame):
            return test_data.values
        elif isinstance(test_data, torch.Tensor):
                return test_data.numpy()
        else:
            raise ValueError("Unsupported data format. Input must be a numpy array or pandas DataFrame.")

    def explain(self, test_data, instance_idx=0):
        """
        Use DLBacktrace to explain a specific instance from the test data.
        :param test_data: Test dataset (numpy array or pandas DataFrame)
        :param instance_idx: Index of the instance to explain
        :return: DataFrame of feature relevance for the explained instance
        """
        test_data = self._prepare_data(test_data)
        instance = test_data[instance_idx:instance_idx+1]

        if self.framework == "pytorch":
            if not isinstance(test_data, torch.Tensor):
                instance = torch.tensor(instance)

        # Step 1: Get the layer outputs using the Backtrace model
        layer_outputs = self.backtrace.predict(instance)
        
        # Step 2: Get the layer-wise relevance using DLBacktrace
        relevance = self.backtrace.eval(
            layer_outputs,
            mode=self.mode,
            scaler=self.scaler,
            thresholding=self.thresholding,
            task=self.task
        )
        # Step 3: Format the relevance output as a DataFrame, focusing on the input layer's relevance
        return self._format_relevance(instance, relevance[self.inp_layer][self.cmode])

    def _format_relevance(self, instance, relevance):
        """
        Format the relevance output into a DataFrame for better readability.
        """
        # Flatten the relevance and instance values to match feature dimensions
        relevance = relevance.flatten()
        instance = instance.flatten()

        # Prepare DataFrame for feature relevance
        relevance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Value': instance,
            'Attribution': relevance
        })

        # Sort by relevance to understand the most influential features
        relevance_df = relevance_df.sort_values(by="Attribution", key=abs, ascending=False)
        return relevance_df
