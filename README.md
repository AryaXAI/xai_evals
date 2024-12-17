# xai_evals

**`xai_evals`** is a Python package designed to generate and benchmark various explainability methods for machine learning and deep learning models. It offers tools for creating and evaluating explanations of popular machine learning models, supporting widely-used explanation methods such as SHAP and LIME. The package aims to streamline the interpretability of machine learning models, allowing practitioners to gain insights into how their models make predictions. Additionally, it includes several metrics for assessing the quality of these explanations.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [SHAP Explainer](#shap-explainer)
  - [LIME Explainer](#lime-explainer)
  - [Metrics Calculation](#metrics-calculation)
- [Extending with More Explanations](#extending-with-more-explanations)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

To install **`xai_evals`**, you can use `pip`. First, clone the repository or download the files to your local environment. Then, install the necessary dependencies:

```bash
git clone https://github.com/AryaXAI/xai_evals.git
cd xai_evals
pip install .
```

Alternatively, if you don't want to clone the repo manually, you can install the package directly from pip (after we publish it [TODO]).

### Dependencies

- `shap`: A library for SHAP values (SHapley Additive exPlanations).
- `lime`: A library for LIME (Local Interpretable Model-Agnostic Explanations).
- `xgboost`: A gradient boosting library.
- `scikit-learn`: For machine learning models and utilities.
- `torch`: For deep learning model support (optional).
- `pandas`: For data handling.
- `numpy`: For numerical computations.

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Usage

Supported Machine Learning Models for `SHAPExplainer` and `LIMEExplainer` class is as follows : 

| **Library**             | **Supported Models**                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------|
| **scikit-learn**         | LogisticRegression, RandomForestClassifier, SVC, SGDClassifier, GradientBoostingClassifier, AdaBoostClassifier, DecisionTreeClassifier, KNeighborsClassifier, GaussianNB, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, KMeans, NearestCentroid, BaggingClassifier, VotingClassifier, MLPClassifier, LogisticRegressionCV, RidgeClassifier, ElasticNet |
| **xgboost**              | XGBClassifier                                                                                         |
| **catboost**             | CatBoostClassifier                                                                                   |
| **lightgbm**             | LGBMClassifier                                                                                       |
| **sklearn.ensemble**     | HistGradientBoostingClassifier, ExtraTreesClassifier                                                  |

### SHAP Explainer

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


### LIME Explainer

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


### TorchExplainer

The `TorchExplainer` class allows you to generate explanations for Pytorch Deep Learning Model . Explaination Method available include 'integrated_gradients', 'deep_lift', 'gradient_shap','saliency', 'input_x_gradient', 'guided_backprop','shap_kernel', 'shap_deep' and 'lime'.

| Attribute    | Description | Values |
|--------------|-------------|--------|
| model | Trained Torch model which you want to explain | [Torch Model] |
| method | Explanation method. Options:'integrated_gradients', 'deep_lift', 'gradient_shap','saliency', 'input_x_gradient', 'guided_backprop','shap_kernel', 'shap_deep', 'lime' | string |
| X_train | Training Set Data | {pd.dataframe,numpy.array} |
| feature_names | Features present in the Training/Testing Set | [list of features] |
| task | Task performed by the model | {binary-classification,multiclass-classification} |

### TFExplainer

The `TFExplainer` class allows you to generate explanations for Tensorflow/Keras Deep Learning Model . Explaination Method available include 'shap_kernel', 'shap_deep' and 'lime'.

| Attribute    | Description | Values |
|--------------|-------------|--------|
| model | Trained Tf/Keras model which you want to explain | [Tf/Keras Model] |
| method | Explanation method. Options:'shap_kernel', 'shap_deep', 'lime' | string |
| X_train | Training Set Data | {pd.dataframe,numpy.array} |
| feature_names | Features present in the Training/Testing Set | [list of features] |
| task | Task performed by the model | {binary-classification,multiclass-classification} |



### BacktraceExplainer

The `BacktraceExplainer` , based on DLBacktrace, a method for analyzing neural networks by tracing the relevance of each component from output to input, to understand how each part contributes to the final prediction. It offers two modes: Default and Contrast, and is compatible with TensorFlow and PyTorch. (https://github.com/AryaXAI/DLBacktrace)
        
| Attribute    | Description | Values |
|--------------|-------------|--------|
| model | Trained Tf/Keras/Torch model which you want to explain | [Torch/Tf/Keras Model] |
| method | Explanation method. Options:"default" or "contrastive" | string |
| X_train | Training Set Data | {pd.dataframe,numpy.array} |
| scaler | Total / Starting Relevance at the Last Layer  | Integer (Default: 1) |
| feature_names | Features present in the Training/Testing Set | [list of features] |
| thresholding | Thresholding for Model Prediction | float (Default : 0.5) |
| task | Task performed by the model | {binary-classification,multiclass-classification} |

### Metrics Calculation

The **`xai_evals`** package provides a powerful class, **`ExplanationMetrics`**, to evaluate the quality of explanations generated by SHAP and LIME. This class allows you to calculate several metrics, helping you assess the robustness, reliability, and interpretability of your model explanations. [NOTE: Metrics only supports Sklearn ML Models]

### ExplanationMetrics Class


The **`ExplanationMetrics`** class in `xai_evals` provides a structured way to evaluate the quality and reliability of explanations generated by SHAP or LIME for machine learning models. By assessing multiple metrics, you can better understand how well these explanations align with your model's predictions and behavior.

---

#### Steps for Using ExplanationMetrics

1. **Initialize ExplanationMetrics**  
   Begin by creating an instance of the `ExplanationMetrics` class with the necessary inputs, including the model, explainer type, dataset, and the task type.

   ```python
   from xai_evals.metrics import ExplanationMetrics
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
   explanation_metrics = ExplanationMetrics(
       model=model,
       explainer_name="shap",
       X_train=X,
       X_test=X,
       y_test=y,
       features=X.columns,
       task="binary"
   )
   ```

For **ExplanationMetrics Class** we have several attributes :


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

#### Benefits of ExplanationMetrics

- **Interpretability:** Quantifies how well feature attributions explain the model's predictions.
- **Robustness:** Evaluates the stability of explanations under input perturbations.
- **Comprehensiveness and Sufficiency:** Provides insights into the contribution of top features to the model’s predictions.
- **Scalability:** Works with various tasks, including binary classification, multi-class classification, and regression.

By leveraging these metrics, you can ensure that your explanations are meaningful, robust, and align closely with your model's decision-making process.

---

## Extending with More Explanations

We plan to expand this library to include more explanation methods, such as:

- **Integrated Gradients**: An attribution method for deep learning models.
- **DeepLIFT**: A method designed for deep learning models that calculates the contribution of each feature to the model's prediction.
- **KernelSHAP**: A kernel-based approximation of SH

AP values that can be applied to any model.

As we add more explanation methods, the `SHAPExplainer` and `LIMEExplainer` classes will be extended to support them, and new classes may be added for other explanation techniques.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Future Plans

In the future, we will continue to improve this library by:

- Adding support for more explanation techniques.
- Enhancing the metrics calculation with more advanced techniques.
- Providing better visualization for SHAP and LIME explanations.
- Improving the documentation and usability of the library.

--- 
