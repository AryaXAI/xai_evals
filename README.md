# xai_evals

**`xai_evals`** is a Python package designed for explainable AI (XAI) and model interpretability. It provides tools for generating and evaluating explanations of machine learning models, with support for popular explanation methods such as **SHAP** and **LIME**. The package aims to simplify the interpretability of machine learning models, enabling practitioners to understand how their models make predictions. It also includes several metrics for evaluating the quality of these explanations, focusing on tabular data.

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
git clone https://github.com/yourusername/xai_evals.git
cd xai_evals
pip install -e .
```

Alternatively, if you don't want to clone the repo manually, you can install the package directly from pip (after publishing it [TODO]).

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
shap_explainer = SHAPExplainer(model=model, features=X.columns, task="classification", X_train=X)

# Explain a specific instance (e.g., the first instance in the test set)
shap_attributions = shap_explainer.explain(X, instance_idx=0)

# Print the feature attributions
print(shap_attributions)
```

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
lime_explainer = LIMEExplainer(model=model, features=X.columns, task="classification", X_train=X)

# Explain a specific instance (e.g., the first instance in the test set)
lime_attributions = lime_explainer.explain(X, instance_idx=0)

# Print the feature attributions
print(lime_attributions)
```

### Metrics Calculation

The **`xai_evals`** package provides a powerful class, **`ExplanationMetrics`**, to evaluate the quality of explanations generated by SHAP and LIME. This class allows you to calculate several metrics, helping you assess the robustness, reliability, and interpretability of your model explanations.

#### ExplanationMetrics Class

### Evaluating ExplanationMetrics

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
| **AUC (Top-k)**      | Evaluates the discriminatory power of the top-k features for classification tasks.            | Calculates the Area Under the Curve (AUC) for top-k features in binary/multi-class tasks.       |
| **Complexity**       | Measures the sparsity of explanations.                                                      | Counts the number of features with non-zero attribution values.                                |
| **Sparseness**       | Assesses how minimal the explanation is.                                                     | Calculates the proportion of features with zero attribution values.                            |

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
| AUC (Top-k)       | 0.92    |
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