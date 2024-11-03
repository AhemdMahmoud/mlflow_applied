# MLflow Experiment Tracking Guide

## Overview
MLflow is an open-source platform for managing the complete machine learning lifecycle. It enables users to log parameters, metrics, and artifacts to keep track of model training processes. This guide covers how to set up MLflow and use it for tracking experiments with models like `LogisticRegression` in scikit-learn and hyperparameter optimization using Optuna.

## Prerequisites
Ensure you have the following installed:
- Python 3.6 or higher
- `mlflow`
- `scikit-learn`
- `optuna`

You can install them using:
```bash
pip install mlflow scikit-learn optuna
```

## Getting Started with MLflow
1. **Set Up MLflow Tracking Server**:
   - You can run MLflow's UI locally to visualize experiment tracking data.
   ```bash
   mlflow ui --port 5000
   ```
   Navigate to `http://localhost:5000` in your browser.

2. **Basic Experiment Logging**:
   Use the following code snippet to log metrics, parameters, and models in MLflow:
   ```python
   import mlflow
   import mlflow.sklearn
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score

   # Load data
   data = load_iris()
   X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

   # Set the experiment
   mlflow.set_experiment("Iris_Classification")

   # Train and log model
   with mlflow.start_run():
       model = LogisticRegression(max_iter=200)
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)

       # Log parameters and metrics
       mlflow.log_param("max_iter", 200)
       mlflow.log_metric("accuracy", accuracy)

       # Log the model
       mlflow.sklearn.log_model(model, "logistic_regression_model")

       print(f"Model accuracy: {accuracy}")
   ```

## Hyperparameter Tuning with Optuna and MLflow
Combine MLflow with Optuna for hyperparameter tuning and automatic experiment logging:
```python
import optuna
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    max_iter = trial.suggest_int("max_iter", 100, 500)
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with mlflow.start_run(nested=True):
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

    return accuracy

# Run the study
mlflow.set_experiment("Optuna_Hyperparameter_Tuning")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Print best trial
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
```

## Accessing MLflow UI
1. Run the MLflow server:
   ```bash
   mlflow ui --port 5000
   ```
2. Open your web browser and navigate to `http://localhost:5000` to explore your logged experiments.

## Conclusion
With MLflow and Optuna, you can track experiments, log parameters, metrics, and model artifacts, making your machine learning workflow more efficient and reproducible.

---

Feel free to customize this guide based on your specific project needs or additional features you wish to highlight!

# Tanks Dr `Eman Raslan`🤷‍♀️✨
