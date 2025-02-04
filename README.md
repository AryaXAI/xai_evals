# xai_evals
## by - AryaXai

**`xai_evals`** is a Python package designed to generate and benchmark various explainability methods for machine learning and deep learning models. It offers tools for creating and evaluating explanations of popular machine learning models, supporting widely-used explanation methods. The package aims to streamline the interpretability of machine learning models, allowing practitioners to gain insights into how their models make predictions. Additionally, it includes several metrics for assessing the quality of these explanations.


![Data Type (2)](https://github.com/user-attachments/assets/134a0b43-0cb8-4ed2-b815-c49e8b0c6c7b)

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [SHAP Tabular Explainer](#shap-tabular-explainer)
  - [LIME Tabular Explainer](#lime-tabular-explainer)
  - [Torch Tabular Explainer](#torch-tabular-explainer)
  - [TFKeras Tabular Explainer](#tfkeras-tabular-explainer)
  - [DlBacktrace Tabular Explainer](#dlbacktrace-tabular-explainer)
  - [Tabular Metrics Calculation](#tabular-metrics-calculation)
  - [Torch Image Explainer](#torch-image-explainer)
  - [TFKeras Image Explainer](#tfkeras-image-explainer)
  - [DlBacktrace Image Explainer](#dlbacktrace-image-explainer)
  - [Tabular Metrics Calculation](#tabular-metrics-calculation)
  - [Image Metrics Calculation](#image-metrics-calculation)
- [License](#license)

---

## Installation

To install **`xai_evals`**, you can use `pip`. First, clone the repository or download the files to your local environment. Then, install the necessary dependencies:

```bash
pip install xai_evals
```

## Usage

Supported Machine Learning Models for `SHAPExplainer` and `LIMEExplainer` class is as follows : 

| **Library**             | **Supported Models**                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------|
| **scikit-learn**         | LogisticRegression, RandomForestClassifier, SVC, SGDClassifier, GradientBoostingClassifier, AdaBoostClassifier, DecisionTreeClassifier, KNeighborsClassifier, GaussianNB, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, KMeans, NearestCentroid, BaggingClassifier, VotingClassifier, MLPClassifier, LogisticRegressionCV, RidgeClassifier, ElasticNet |
| **xgboost**              | XGBClassifier                                                                                         |
| **catboost**             | CatBoostClassifier                                                                                   |
| **lightgbm**             | LGBMClassifier                                                                                       |
| **sklearn.ensemble**     | HistGradientBoostingClassifier, ExtraTreesClassifier                                                  |

### SHAP Tabular Explainer

The `SHAPExplainer` class allows you to compute and visualize **SHAP** values for your trained model. It supports various types of models, including tree-based models (e.g., `RandomForest`, `XGBoost`) and deep learning models (e.g., PyTorch models).

**Example:**

```python
from xai_evals.explainer import SHAPExplainer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.datasets import load_iris

# Load dataset and train a model
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
model = RandomForestClassifier()
model.fit(X, y)

# Initialize SHAP explainer
shap_explainer = SHAPExplainer(model=model, features=X.columns, task="multiclass-classification", X_train=X)

# Explain a specific instance (e.g., the first instance in the test set)
shap_attributions = shap_explainer.explain(X, instance_idx=0)

# Print the feature attributions
print(shap_attributions)
```

| **Feature**           | **Value** | **Attribution** |
|-----------------------|-----------|-----------------|
| petal_length_(cm)     | 1.4       | 0.360667        |
| petal_width_(cm)      | 0.2       | 0.294867        |
| sepal_length_(cm)     | 5.1       | 0.023467        |
| sepal_width_(cm)      | 3.5       | 0.010500        |


### LIME Tabular Explainer

The `LIMEExplainer` class allows you to generate **LIME** explanations, which work by perturbing the input data and fitting a locally interpretable model.

**Example:**

```python
from xai_evals.explainer import LIMEExplainer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.datasets import load_iris

# Load dataset and train a model
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Initialize LIME explainer
lime_explainer = LIMEExplainer(model=model, features=X.columns, task="multiclass-classification", X_train=X)

# Explain a specific instance (e.g., the first instance in the test set)
lime_attributions = lime_explainer.explain(X, instance_idx=0)

# Print the feature attributions
print(lime_attributions)
```
| **Feature**           | **Value** | **Attribution** |
|-----------------------|-----------|-----------------|
| petal_length_(cm)     | 1.4       | 0.497993        |
| petal_width_(cm)      | 0.2       | 0.213963        |
| sepal_length_(cm)     | 5.1       | 0.127047        |
| sepal_width_(cm)      | 3.5       | 0.053926        |

For **LIMEExplainer and SHAPExplainer Class** we have several attributes :

| Attribute    | Description | Values |
|--------------|-------------|--------|
| model | Trained model which you want to explain | [sklearn model] |
| features | Features present in the Training/Testing Set | [list of features] |
| X_train | Training Set Data | {pd.dataframe,numpy.array} |
| task | Task performed by the model | {binary,multiclass} |
| model_classes (Only for LIME) | List of Classes to be predicted by model | [list of classes] |
| subset_samples (Only for SHAP) | If we want to use k-means based sampling to use a subset for SHAP Explainer | True/False |
| subset_number (Only for SHAP)| Number of samples to sample if subset_samples is True | int |


### Torch Tabular Explainer

The `TorchTabularExplainer` class allows you to generate explanations for Pytorch Deep Learning Model . Explaination Method available include 'integrated_gradients', 'deep_lift', 'gradient_shap','saliency', 'input_x_gradient', 'guided_backprop','shap_kernel', 'shap_deep' and 'lime'.

| Attribute    | Description | Values |
|--------------|-------------|--------|
| model | Trained Torch model which you want to explain | [Torch Model] |
| method | Explanation method. Options:'integrated_gradients', 'deep_lift', 'gradient_shap','saliency', 'input_x_gradient', 'guided_backprop','shap_kernel', 'shap_deep', 'lime' | string |
| X_train | Training Set Data | {pd.dataframe,numpy.array} |
| feature_names | Features present in the Training/Testing Set | [list of features] |
| task | Task performed by the model | {binary-classification,multiclass-classification} |

### TFKeras Tabular Explainer

The `TFTabularExplainer` class allows you to generate explanations for Tensorflow/Keras Deep Learning Model . Explaination Method available include 'shap_kernel', 'shap_deep' and 'lime'.

| Attribute    | Description | Values |
|--------------|-------------|--------|
| model | Trained Tf/Keras model which you want to explain | [Tf/Keras Model] |
| method | Explanation method. Options:'shap_kernel', 'shap_deep', 'lime' | string |
| X_train | Training Set Data | {pd.dataframe,numpy.array} |
| feature_names | Features present in the Training/Testing Set | [list of features] |
| task | Task performed by the model | {binary-classification,multiclass-classification} |



### DlBacktrace Tabular Explainer

The `DlBacktraceTabularExplainer` , based on DLBacktrace, a method for analyzing neural networks by tracing the relevance of each component from output to input, to understand how each part contributes to the final prediction. It offers two modes: Default and Contrast, and is compatible with TensorFlow and PyTorch. (https://github.com/AryaXAI/DLBacktrace)
        
| Attribute    | Description | Values |
|--------------|-------------|--------|
| model | Trained Tf/Keras/Torch model which you want to explain | [Torch/Tf/Keras Model] |
| method | Explanation method. Options:"default" or "contrastive" | string |
| X_train | Training Set Data | {pd.dataframe,numpy.array} |
| scaler | Total / Starting Relevance at the Last Layer  | Integer (Default: 1) |
| feature_names | Features present in the Training/Testing Set | [list of features] |
| thresholding | Thresholding for Model Prediction | float (Default : 0.5) |
| task | Task performed by the model | {binary-classification,multiclass-classification} |

### Torch Image Explainer

The `TorchImageExplainer` class allows you to generate explanations for PyTorch-based CNN models. This class wraps around several attribution methods available in Captum, including:

- **Integrated Gradients**
- **Saliency**
- **DeepLift**
- **GradientShap**
- **GuidedBackprop**
- **Occlusion**
- **LayerGradCam**

**Example:**

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from xai_evals.explainer import TorchImageExplainer
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Initialize the TorchImageExplainer
explainer = TorchImageExplainer(model)

# Example 1: Explain using DataLoader (batch of images)
idx = 0  # Index for the image in the DataLoader
method = "integrated_gradients"
task = "classification"
attribution_map = explainer.explain(trainloader, idx, method, task)

# Visualize attribution map (simplified)
plt.imshow(attribution_map)
plt.title(f"Attribution Map - {method} for Dataloader Torch")
plt.show()

# Example 2: Explain using a single image (torch.Tensor)
single_image_tensor = torch.randn(3, 32, 32)  # Random image as a tensor, [C, H, W]
attribution_map = explainer.explain(single_image_tensor, idx=None, method=method, task=task)

# Visualize attribution map for the single image
plt.imshow(attribution_map)
plt.title(f"Attribution Map - {method} for Single Image (Tensor)")
plt.show()

# Example 3: Explain using a single image (np.ndarray)
single_image_numpy = np.random.randn(3, 32, 32)  # Random image as a NumPy array, [C, H, W]
attribution_map = explainer.explain(single_image_numpy, idx=None, method=method, task=task)

# Visualize attribution map for the single image (NumPy)
plt.imshow(attribution_map)
plt.title(f"Attribution Map - {method} for Single Image (NumPy)")
plt.show()
```

#### **TorchImageExplainer**: `explain` Function Attributes

| **Attribute** | **Description** | **Values** |
|---------------|-----------------|-----------|
| `testdata`    | The input data, which can be a DataLoader, NumPy array, or Tensor. | `[torch.utils.data.DataLoader, np.ndarray, torch.Tensor]` |
| `idx`         | The index of the test sample to explain. | `int` or `None` (for explaining a single sample or all samples) |
| `method`      | The explanation method to use. | `{grad_cam, integrated_gradients, saliency, deep_lift, gradient_shap, guided_backprop, occlusion, layer_gradcam, feature_ablation}` |
| `task`        | The type of model task (e.g., classification). | `{classification}` |

---

### TFKeras Image Explainer

The `TFImageExplainer` class provides a similar functionality for TensorFlow/Keras-based models, allowing you to generate explanations for images using methods like GradCAM and Occlusion Sensitivity.

**Example:**

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from xai_evals.explainer import TFImageExplainer

# Step 1: Define a Custom CNN Model
def create_custom_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Create a TensorFlow Dataset from the test data
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Initialize and train the custom model
model = create_custom_model()

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=64)

# Step 2: Use the TFImageExplainer with the Custom Model
explainer = TFImageExplainer(model)

# Example 1: Explain a Single Image (NumPy Array)
image = x_test[0]  # Select the first image
label = y_test[0]  # Get the label for the first image

# Generate the Grad-CAM explanation for the image
attribution_map = explainer.explain(image, idx=None, method="grad_cam", task="classification")

# Visualize the attribution map
plt.imshow(attribution_map, cmap="jet")
plt.colorbar()
plt.title("Grad-CAM Attribution Map for CIFAR-10 Image")
plt.show()

# Example 2: Explain an Image from the TensorFlow Dataset (Using idx)
idx = 10  # Select the 10th image from the test dataset

# Generate the Grad-CAM explanation for the image at index `idx`
attribution_map = explainer.explain(test_dataset, idx, method="grad_cam", task="classification")

# Visualize the attribution map
plt.imshow(attribution_map, cmap="jet")
plt.colorbar()
plt.title(f"Grad-CAM Attribution Map for Image Index {idx} in CIFAR-10")
plt.show()
```

#### **TFImageExplainer**: `explain` Function Attributes

| **Attribute** | **Description** | **Values** |
|---------------|-----------------|-----------|
| `testset`     | The input data, which can be a NumPy array, TensorFlow tensor, or Dataset. | `[np.ndarray, tf.Tensor, tf.data.Dataset]` |
| `idx`         | The index of the test sample to explain. | `int` or `None` (for explaining a single sample or all samples) |
| `method`      | The explanation method to use. | `{grad_cam, occlusion}` |
| `task`        | The type of model task (e.g., classification). | `{classification}` |
| `label`       | The class label for the input sample (used for classification tasks). | `int` |

---

### DlBacktrace Image Explainer

The `DlBacktraceImageExplainer` based on DLBacktrace, a method for analyzing neural networks by tracing the relevance of each component from output to input, to understand how each part contributes to the final prediction. It offers two modes: Default and Contrast, and is compatible with TensorFlow and PyTorch. (https://github.com/AryaXAI/DLBacktrace)

**Example: Tensorflow Model DlBacktraceImageExplainer**

```python
# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a simple CNN model for CIFAR-10
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Create the model
model = create_cnn_model()

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the model for later use
model.save('cifar10_cnn_model.h5')

explainer = BacktraceImageExplainer(model=model)

# Choose an image from the test set
test_image = x_test[0:1]  # Selecting the first image for testing

# Get the explanation for the test image
explanation = explainer.explain(test_image, instance_idx=0,mode='default', scaler=1, thresholding=0, task='multi-class-classification')

# Plot the explanation (relevance map)
plt.imshow(explanation, cmap='hot')
plt.colorbar()
plt.title("Feature Relevance for CIFAR-10 Image")
plt.show()
```

**Example: Torch Model DlBacktraceImageExplainer**

```python
# Define a simple CNN model for CIFAR-10 without using `view()`
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.identity = nn.Identity()
        self.conv1 = nn.Conv2d(3, 16, 5,2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3,2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 6 * 6, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.identity(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Load CIFAR-10 data with transforms for normalization
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
testloader = DataLoader(testset, batch_size=4, shuffle=False)

# Initialize and train the model
model = SimpleCNN()
model.train()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(1):  # Just a couple of epochs for testing
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

print("Finished Training")

# Test the model using the BacktraceImageExplainer
explainer = DlBacktraceImageExplainer(model=model)


# Get the explanation for the first image
explanation = explainer.explain(testloader, instance_idx=0, mode='default', scaler=1, thresholding=0, task='multi-class-classification')

# Plot the explanation (relevance map)
plt.imshow(explanation, cmap='hot')
plt.colorbar()
plt.title("Feature Relevance for CIFAR-10 Image")
plt.show()
```

#### **DlBacktraceImageExplainer**: `explain` Function Attributes

| **Attribute** | **Description** | **Values** |
|---------------|-----------------|-----------|
| `test_data`     | The input data, which can be a NumPy array, TensorFlow tensor, or Dataset. | `[np.ndarray, tf.Tensor, tf.data.Dataset]` |
| `instance_idx`         | The index of the test sample to explain. | `int` (explaining a single sample) |
| `mode`      | The explanation mode to use. | `{default, contrast}` |
| `task`        | The type of model task (e.g., classification). | `{binary-classification,multiclass-classification}` |
| `scaler`       | Total / Starting Relevance at the Last Layer	 | `float` ( Default: None, Preferred: 1) |
| `thresholding` | Thresholding Model Prediction to predict the actual class. | `float` |
| `contrast_mode` | Mode to Use if using 'contrast' mode of DlBacktrace Algorithm | `{Positive,Negative}` |

---

### Tabular Metrics Calculation

The **`xai_evals`** package provides a powerful class, **`ExplanationMetricsTabular`**, to evaluate the quality of explanations generated by SHAP and LIME. This class allows you to calculate several metrics, helping you assess the robustness, reliability, and interpretability of your model explanations. [NOTE: Metrics only supports Sklearn ML Models]

#### ExplanationMetrics Class


The **`ExplanationMetricsTabular`** class in `xai_evals` provides a structured way to evaluate the quality and reliability of explanations generated by SHAP or LIME for machine learning models. By assessing multiple metrics, you can better understand how well these explanations align with your model's predictions and behavior.

---

#### Steps for Using ExplanationMetrics

1. **Initialize ExplanationMetrics**  
   Begin by creating an instance of the `ExplanationMetricsTabular` class with the necessary inputs, including the model, explainer type, dataset, and the task type.

   ```python
   from xai_evals.metrics import ExplanationMetricsTabular
   from xai_evals.explainer import SHAPExplainer
   from sklearn.ensemble import RandomForestClassifier
   import pandas as pd
   from sklearn.datasets import load_iris

   # Load dataset and train a model
   data = load_iris()
   X = pd.DataFrame(data.data, columns=data.feature_names)
   y = data.target
   model = RandomForestClassifier()
   model.fit(X, y)

   # Initialize ExplanationMetrics with SHAP explainer
   explanation_metrics = ExplanationMetricsTabular(
       model=model,
       explainer_name="shap",
       X_train=X,
       X_test=X,
       y_test=y,
       features=X.columns,
       task="binary"
   )
   ```

For **ExplanationMetricsTabular Class** we have several attributes :


| Attribute    | Description | Values |
|--------------|-------------|--------|
| model | Trained model which you want to explain | {binary-classification, multiclass-classification}|
| X_train | Training Set Data | {pd.dataframe,numpy.array} |
| explainer_name | Which explaination method to use | {'shap','lime','torch','tensorflow', 'backtrace'} |
| X_test | Test Set Data | {pd.dataframe,numpy.array} |
| y_test | Test Set Labels | pd.dataseries |
| features | Features present in the Training/Testing Set | [list of features] |
| task | Task performed by the model | {binary-classification,multiclass-classification} |
| metrics | List of metrics to calculate | ['faithfulness', 'infidelity', 'sensitivity', 'comprehensiveness', 'sufficiency', 'monotonicity', 'complexity', 'sparseness'] |
| method | For specifying which explaination Method to use in Torch/Tensorflow/Backtrace Explainer | Torch-{ 'integrated_gradients', 'deep_lift', 'gradient_shap','saliency', 'input_x_gradient', 'guided_backprop','shap_kernel', 'shap_deep','lime'}, Tensorflow-{'shap_kernel','shap_deep','lime'},Backtrace-{'Default','Contrastive'} |
| start_idx | Starting index of the dataset to evaluate | int |
| end_idx |  Ending index of the dataset to evaluate | int |
| scaler | Total / Starting Relevance at the Last Layer	Integer ( For Backtrace) | int (Default: None, Preferred: 1) |
|thresholding | Thresholding Model Prediction | float (default=0.5) |
|subset_samples | If we want to use k-means based sampling to use a subset for SHAP Explainer (Only for SHAP) |	True/False |
|subset_number | Number of samples to sample if subset_samples is True (Only for SHAP) |	int |


2. **Calculate Explanation Metrics**  
   Use the `calculate_metrics` method to compute various metrics for evaluating explanations. The method returns a DataFrame with the results.

   ```python
   # Calculate metrics
   metrics_df = explanation_metrics.calculate_metrics()
   print(metrics_df)
   ```

---

#### Explanation Metrics Overview

The **`ExplanationMetrics`** class supports the following key metrics for evaluating explanations:

| **Metric**          | **Purpose**                                                                                  | **Description**                                                                                 |
|----------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| **Faithfulness**     | Measures consistency between attributions and prediction changes.                            | Correlation between attribution values and changes in model output when features are perturbed. |
| **Infidelity**       | Assesses how closely attributions align with the actual prediction impact.                   | Squared difference between predicted and actual impact when features are perturbed.            |
| **Sensitivity**      | Evaluates the robustness of attributions to small changes in inputs.                         | Compares attribution values before and after perturbing input features.                        |
| **Comprehensiveness**| Assesses the explanatory power of the top-k features.                                        | Measures how much model prediction decreases when top-k important features are removed.         |
| **Sufficiency**      | Determines whether top-k features alone are sufficient to explain the model's output.        | Compares predictions based only on the top-k features to baseline predictions.                 |
| **Monotonicity**     | Verifies the consistency of attribution values with the direction of predictions.             | Ensures that changes in attributions match consistent changes in predictions.                  |
| **Complexity**       | Measures the sparsity of explanations.                                                      | Counts the number of features with non-zero attribution values.                                |
| **Sparseness**       | Assesses how minimal the explanation is.                                                     | Calculates the proportion of features with zero attribution values.                            |

Reference Values for Available Metrics : 

| Metric           | Typical Range            | Interpretation                                                                                                | "Better" Direction                  |
|------------------|--------------------------|---------------------------------------------------------------------------------------------------------------|-------------------------------------|
| Faithfulness      | -1 to 1                 | Measures correlation between attributions and changes in model output when removing features. Higher indicates that more important features (according to the explanation) indeed cause larger changes in the model’s prediction. | Higher is better (closer to 1)      |
| Infidelity        | ≥ 0                     | Measures how well attributions predict changes in the model’s output under input perturbations. Lower infidelity means the attributions closely match the model’s behavior under perturbations. | Lower is better (closer to 0)       |
| Sensitivity       | ≥ 0                     | Measures how stable attributions are to small changes in the input. Lower values mean more stable (robust) explanations. | Lower is better (closer to 0)       |
| Comprehensiveness | Depends on model output | Measures how much the prediction drops when the top-k most important features are removed. If removing them significantly decreases the prediction, it suggests these features are truly important. | Higher difference indicates more comprehensive explanations |
| Sufficiency       | Depends on model output | Measures how well the top-k features alone approximate or even match the original prediction. A higher (or less negative) value means these top-k features are sufficient on their own, capturing most of what the model uses. | Higher (or closer to zero if baseline is the original prediction) is generally better |
| Monotonicity      | 0 to 1 (as an average)  | Checks if attributions are in a non-increasing order. A higher average indicates that the explanation presents a consistent ranking of feature importance. | Higher is better (closer to 1)      |
| Complexity        | Depends on number of features | Measures the number of non-zero attributions. More features with non-zero attributions mean a more complex explanation. Fewer important features make it easier to interpret. | Lower is typically preferred        |
| Sparseness        | 0 to 1                  | Measures the fraction of attributions that are zero. Higher sparseness means fewer features are highlighted, making the explanation simpler. | Higher is generally preferred       |

---

#### Practical Examples

**1. Faithfulness Correlation**
   - Correlates feature attributions with prediction changes when features are perturbed. 
   - Higher correlation indicates that the explanation aligns well with model predictions.

   ```python
   faithfulness_score = explanation_metrics.calculate_metrics()['faithfulness']
   print("Faithfulness:", faithfulness_score)
   ```

**2. Infidelity**
   - Computes the squared difference between predicted and actual changes in model output.
   - Lower scores indicate higher alignment of explanations with model behavior.

   ```python
   infidelity_score = explanation_metrics.calculate_metrics()['infidelity']
   print("Infidelity:", infidelity_score)
   ```

**3. Comprehensiveness**
   - Evaluates whether removing the top-k features significantly reduces the model's prediction confidence.
   - A higher score indicates that the top-k features are critical for the prediction.

   ```python
   comprehensiveness_score = explanation_metrics.calculate_metrics()['comprehensiveness']
   print("Comprehensiveness:", comprehensiveness_score)
   ```

---

#### Example Output

After calculating the metrics, the method returns a DataFrame summarizing the results:

| Metric           | Value   |
|-------------------|---------|
| Faithfulness      | 0.89    |
| Infidelity        | 0.05    |
| Sensitivity       | 0.13    |
| Comprehensiveness | 0.62    |
| Sufficiency       | 0.45    |
| Monotonicity      | 1.00    |
| Complexity        | 7       |
| Sparseness        | 0.81    |

---

### Image Metrics Calculation

The **`xai_evals`** package provides a powerful class, **`ExplanationMetricsImage`**, to evaluate the quality of explanations generated for image-based deep learning models. This class allows you to calculate several metrics, helping you assess the robustness, reliability, and interpretability of your image explanations. [NOTE: Metrics currently support image-based deep learning models such as PyTorch and TensorFlow.]

#### ExplanationMetricsImage Class

The **`ExplanationMetricsImage`** class in **`xai_evals`** provides a structured way to evaluate the quality and reliability of image-based explanations, such as GradCAM, Integrated Gradients, and Occlusion. By assessing multiple metrics, you can better understand how well these image explanations align with your model's predictions and behavior. This class uses **Quantus** to calculate the various metrics for evaluating explanations.

---

#### Steps for Using ExplanationMetricsImage

1. **Initialize ExplanationMetricsImage**  
   Begin by creating an instance of the **`ExplanationMetricsImage`** class with the necessary inputs, including the model, dataset, and evaluation settings.

2. **Evaluate Explanation Metrics**  
   Use the `evaluate` method to compute various metrics for evaluating image-based explanations. The method returns a dictionary with the results.

   ```python
   import torch
   import torchvision
   import torchvision.transforms as transforms
   from torch.utils.data import DataLoader
   from tensorflow.keras.datasets import cifar10
   from xai_evals.metrics import ExplanationMetricsImage
   from torchvision import models
   import tensorflow as tf
   import numpy as np
   import torch.optim as optim
   import tensorflow.keras as keras

   # --- TensorFlow Setup ---
   # Load CIFAR-10 dataset (for TensorFlow example)
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the images
   train_data = (x_train, y_train)  # Tuple of data and labels
   test_data = (x_test, y_test)     # Tuple of data and labels

   # Convert to TensorFlow Dataset
   train_dataset_tf = tf.data.Dataset.from_tensor_slices(train_data).batch(32)
   test_dataset_tf = tf.data.Dataset.from_tensor_slices(test_data).batch(32)

   # --- PyTorch Setup ---
   # PyTorch Dataset for CIFAR-10
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
   trainloader = DataLoader(trainset, batch_size=4, shuffle=True)


   # --- Custom Model Setup ---
   # Custom PyTorch model (simple CNN for CIFAR-10)
   class SimpleCNN(torch.nn.Module):
      def __init__(self):
         super(SimpleCNN, self).__init__()
         self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
         self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
         self.fc1 = torch.nn.Linear(64*8*8, 128)
         self.fc2 = torch.nn.Linear(128, 10)  # 10 classes for CIFAR-10

      def forward(self, x):
         x = torch.relu(self.conv1(x))
         x = torch.max_pool2d(x, 2)
         x = torch.relu(self.conv2(x))
         x = torch.max_pool2d(x, 2)
         x = x.view(x.size(0), -1)
         x = torch.relu(self.fc1(x))
         x = self.fc2(x)
         return x

   # --- TensorFlow Model Setup ---
   model_tf = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)  # 10 classes for CIFAR-10
   ])

   # Compile the model for training
   model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   # --- TensorFlow Model Training (1 Epoch) ---
   model_tf.fit(train_dataset_tf, epochs=1)

   print("Finished TensorFlow Training")
   # Initialize PyTorch model
   model_torch = SimpleCNN()
   model_torch.train()  # Set model to training mode

   # --- Training PyTorch Model for 1 Epoch ---
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = optim.SGD(model_torch.parameters(), lr=0.001, momentum=0.9)

   for epoch in range(1):  # Training for 1 epoch
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
         inputs, labels = data
         optimizer.zero_grad()

         outputs = model_torch(inputs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()

         running_loss += loss.item()
         if i % 2000 == 1999:  # Print every 2000 mini-batches
               print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
               running_loss = 0.0

   print("Finished PyTorch Training")
   # --- Example 1: PyTorch Metrics Calculation ---
   metrics_image_pytorch = ExplanationMetricsImage(
      model=model_torch, 
      data_loader=trainloader, 
      framework="torch", 
      num_classes=10
   )

   # Example: Calculate metrics using the PyTorch DataLoader
   metrics_results_pytorch = metrics_image_pytorch.evaluate(
      start_idx=0, end_idx=32, 
      metric_names=["FaithfulnessCorrelation","MaxSensitivity","MPRT","SmoothMPRT","AvgSensitivity","FaithfulnessEstimate"], 
      xai_method_name="IntegratedGradients"
   )
   print("PyTorch Example Metrics:", metrics_results_pytorch)
   # --- Example 2: TensorFlow Metrics Calculation ---
   metrics_image_tensorflow = ExplanationMetricsImage(
      model=model_tf,  # Use TensorFlow model for TensorFlow example
      data_loader=train_dataset_tf,
      framework="tensorflow",
      num_classes=10
   )

   # Example: Calculate metrics using the TensorFlow Dataset
   metrics_results_tensorflow = metrics_image_tensorflow.evaluate(
      start_idx=0, end_idx=32, 
      metric_names=["FaithfulnessCorrelation","MaxSensitivity","MPRT","SmoothMPRT","AvgSensitivity","FaithfulnessEstimate"],
      xai_method_name="GradCAM"
   )
   print("TensorFlow Example Metrics:", metrics_results_tensorflow)
   # --- Example 3: Explain using a single image (numpy array) ---
   single_image_numpy = np.random.randn(1,3,32, 32)  # Random image as a NumPy array, [H, W, C]
   label = np.random.randint(0, 9,size=1)

   # Initialize ExplanationMetricsImage for a single image (use PyTorch framework even for NumPy array)
   metrics_image_single = ExplanationMetricsImage(
      model=model_torch,  # Use PyTorch model
      data_loader=(single_image_numpy,label),  # Pass the single image as a numpy array
      framework="torch",  # Use the torch framework for single image
      num_classes=10,
   )

   # Calculate metrics for the single image
   metrics_single_image = metrics_image_single.evaluate(
      start_idx=0, end_idx=1, 
      metric_names=["FaithfulnessCorrelation","MaxSensitivity","MPRT","SmoothMPRT","AvgSensitivity","FaithfulnessEstimate"],
      xai_method_name="IntegratedGradients"
   )
   print("Single Image Example Metrics:", metrics_single_image)
   # --- Example 4: TensorFlow Model with Single Image ---
   single_image_numpy = np.random.randn(1,32, 32,3)  # Random image as a NumPy array, [H, W, C]
   label = np.random.randint(0, 9,size=1)
   # For TensorFlow, the single image example using TensorFlow framework
   metrics_image_single_tf = ExplanationMetricsImage(
      model=model_tf,  # Use TensorFlow model
      data_loader=(single_image_numpy,label),  # Pass the single image as a numpy array
      framework="tensorflow",  # Use the tensorflow framework for single image
      num_classes=10
   )

   # Calculate metrics for the single image
   metrics_single_image_tf = metrics_image_single_tf.evaluate(
      start_idx=0, end_idx=1, 
      metric_names=["FaithfulnessCorrelation","MaxSensitivity","MPRT","SmoothMPRT","AvgSensitivity","FaithfulnessEstimate"],
      xai_method_name="GradCAM"
   )
   print("TensorFlow Single Image Example Metrics:", metrics_single_image_tf)


   ```

---

#### Explanation Metrics Overview

The **`ExplanationMetricsImage`** class supports the following key metrics for evaluating image explanations:

| **Metric**               | **Purpose**                                                                                     | **Description**                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **FaithfulnessCorrelation** | Measures the correlation between attribution values and model output changes when perturbing image features. | Higher values indicate that important features (according to the explanation) indeed cause significant changes in the model’s prediction. |
| **MaxSensitivity**        | Measures the maximum sensitivity of an attribution method to input perturbations.                | Higher values suggest that the attribution method highlights the most sensitive parts of the image.       |
| **MPRT**                  | Measures the relevance of features based on perturbations.                                       | Helps evaluate the robustness of the explanation when features are perturbed.                              |
| **SmoothMPRT**            | A smoother version of MPRT that reduces noise from perturbations.                                | Ensures more stable results by averaging perturbations.                                                   |
| **AvgSensitivity**        | Measures the average sensitivity of the model to input perturbations across all features.        | Indicates how sensitive the model is to small changes in the input.                                       |
| **FaithfulnessEstimate**  | Estimates the faithfulness of the attribution by comparing against a perturbation baseline.     | Compares how well the explanation reflects the model’s behavior under feature perturbations.               |

Reference Values for Available Metrics:

| Metric                   | Typical Range           | Interpretation                                                                                           | "Better" Direction                   |
|--------------------------|-------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------|
| FaithfulnessCorrelation   | -1 to 1                 | Measures correlation between attribution values and changes in model output when features are perturbed. Higher indicates that more important features (according to the explanation) indeed cause larger changes in the model’s prediction. | Higher is better (closer to 1)       |
| MaxSensitivity            | ≥ 0                     | Measures how well attributions match model sensitivity when perturbing image features. Lower scores indicate that the explanations focus on the most sensitive features. | Lower is better (closer to 0)        |
| MPRT                      | ≥ 0                     | Measures how the perturbation of features affects the model’s prediction. A higher score indicates that the model's prediction is heavily influenced by the perturbed features. | Higher is better                    |
| SmoothMPRT                | ≥ 0                     | Measures the stability of MPRT under perturbation noise. Higher values suggest more stable explanations. | Higher is better                    |
| AvgSensitivity            | ≥ 0                     | Measures the average change in prediction for small changes in input features. Indicates model robustness. | Lower is better                     |
| FaithfulnessEstimate      | 0 to 1                   | Compares model predictions under perturbations and attributions. Higher values indicate better alignment. | Higher is better                    |

---

#### Initialization Attributes / Contructor for **`ExplanationMetricsImage`** class

| **Attribute**        | **Description**                                                                                          | **Values**                                                                                           |
|----------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `model`              | The trained model for which explanations will be evaluated.                                                | [PyTorch model, TensorFlow model]                                                                    |
| `data_loader`        | The data loader or dataset containing the test data.                                                      | [PyTorch Dataset,PyTorch DataLoader,TensorFlow Dataset, tuple of (image-np.array/torch.Tensor/tensorflow.Tensor.Tensor,label-np.array/torch.Tensor/tensorflow.Tensor)]                                                |
| `framework`          | The framework used for the model (either 'torch' or 'tensorflow' or 'backtrace').                                         | {'torch', 'tensorflow','backtrace'}                                                                               |
| `device`             | The device (CPU/GPU) used for performing computations (for PyTorch models).                               | [torch.device (Optional)]                                                                             |
| `num_classes`        | The number of classes for classification tasks.                                                           | Integer (default: 10)                                                                                 |

---

#### Evaluate Function (`evaluate`) for **`ExplanationMetricsImage`** class to calculate metrics

| **Attribute**         | **Description**                                                                                          | **Values**                                                                                           |
|-----------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `start_idx`           | The starting index of the batch for evaluation.                                                           | Integer (e.g., 0)                                                                                   |
| `end_idx`             | The ending index of the batch for evaluation.                                                             | Integer (e.g., 100, or `None` for the entire batch)                                                  |
| `metric_names`        | The list of metric names to evaluate.                                                                     | List of strings representing the metrics to compute (e.g., `["FaithfulnessCorrelation", "MaxSensitivity", "MPRT", "SmoothMPRT", "AvgSensitivity", "FaithfulnessEstimate"]`)        |
| `xai_method_name`     | The name of the XAI method used for explanations (e.g., 'IntegratedGradients', 'GradCAM', etc.).           | String (e.g., for Torch `{grad_cam, integrated_gradients, saliency, deep_lift, gradient_shap, guided_backprop, occlusion, layer_gradcam, feature_ablation}` ; for Tensorflow `{VanillaGradients, GradCAM,GradientsInput,IntegratedGradients,OcclusionSensitivity,SmoothGrad` & for Backtrace `{default,contrast-positive,contrast-negative}`)                                                      |

---



#### Practical Examples

**1. Faithfulness Correlation**
   - Correlates feature attributions with prediction changes when features (pixels) in the image are perturbed.
   - Higher correlation indicates that the explanation aligns well with model predictions.

   ```python
   faithfulness_score = metrics_image.evaluate(
       start_idx=0, end_idx=5, metric_names=["FaithfulnessCorrelation"], xai_method_name="IntegratedGradients"
   )['FaithfulnessCorrelation']
   print("Faithfulness:", faithfulness_score)
   ```

**2. Max Sensitivity**
   - Measures the sensitivity of the explanation method by observing the effect of perturbing different parts of the image.
   - A higher score indicates that the explanation method is sensitive to the most influential pixels.

   ```python
   max_sensitivity_score = metrics_image.evaluate(
       start_idx=0, end_idx=5, metric_names=["MaxSensitivity"], xai_method_name="IntegratedGradients"
   )['MaxSensitivity']
   print("Max Sensitivity:", max_sensitivity_score)
   ```

---

#### Example Output

After calculating the metrics, the method returns a dictionary summarizing the results:

| Metric                   | Value   |
|--------------------------|---------|
| FaithfulnessCorrelation   | 0.88    |
| MaxSensitivity            | 0.92    |

---

#### Benefits of ExplanationMetrics

- **Interpretability:** Quantifies how well feature attributions explain the model's predictions.
- **Robustness:** Evaluates the stability of explanations under input perturbations.
- **Comprehensiveness and Sufficiency:** Provides insights into the contribution of top features to the model’s predictions.
- **Scalability:** Works with various tasks, including binary classification, multi-class classification, and regression.

By leveraging these metrics, you can ensure that your explanations are meaningful, robust, and align closely with your model's decision-making process.

---

### Acknowledgements

We would like to extend our heartfelt thanks to the developers and contributors of the libraries **[Quantus](https://github.com/Trusted-AI/quantus)**, **[Captum](https://captum.ai/)**, **[tf-explain](https://github.com/sicara/tf-explain)**, **[LIME](https://github.com/marcotcr/lime)**, and **[SHAP](https://github.com/slundberg/shap)**, which have been instrumental in enabling the explainability methods implemented in this package.

- **[Quantus](https://github.com/Trusted-AI/quantus)** provides a comprehensive suite of metrics that allow us to evaluate and assess the quality of explanations, ensuring that our interpretability methods are both reliable and robust.
  
- **[Captum](https://captum.ai/)** is an invaluable tool for PyTorch users, offering a variety of powerful attribution methods like Integrated Gradients, Saliency, and Gradient Shap, which are crucial for generating insights into the inner workings of deep learning models.

- **[tf-explain](https://github.com/sicara/tf-explain)** simplifies the process of explaining TensorFlow/Keras models, with methods like GradCAM and Occlusion Sensitivity, enabling us to generate visual explanations that help interpret the decision-making of complex models.

- **[LIME](https://github.com/marcotcr/lime)** (Local Interpretable Model-Agnostic Explanations) has been a key library for providing local explanations for machine learning models, allowing us to generate understandable explanations for individual predictions.

- **[SHAP](https://github.com/slundberg/shap)** (SHapley Additive exPlanations) is essential for computing Shapley values and provides a unified approach to explaining machine learning models, making it easier to understand feature contributions across a range of model types.

We are deeply grateful for the contributions these libraries have made in advancing model interpretability, and their seamless integration in our package ensures that users can leverage state-of-the-art methods for understanding machine learning and deep learning models.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Future Plans

In the future, we will continue to improve this library.



--- 
