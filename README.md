# MLflow Practice session for Agile Data Science class

This small project shows a small use case for [MlFlow](https://mlflow.org/)

## Create a GitHub repository

As seen on class create a new Github Repository

Name it `ub-agile-data-science-mlflow`.

Clone the repository locally via the following command:

- note:substitute `${YOUR_USER}` for your username on GitHub (you can copy and paste the URL from GitHub)

```bash
git clone git@github.com:${YOUR_USER}/ub-agile-data-science-mlflow.git
```

## Repository structure setup:

With your text editor or your IDE (Visual Studio Code for example) create a file `requirements.txt` with:

```
mlflow
```

Create a Dockerfile with the following contents:

```
FROM python:3.13

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
```

Test your Dockerfile via:

```
docker build -t ub-agile-data-science-mlflow .
docker run -p 5000:5000 ub-agile-data-science-mlflow:latest
```

Go to your browser:
http://localhost:5000/

Add, commit and push the requirements.txt file to GitHub.

```bash
git add .
git commit -m "Add initial requirements.txt and Dockerfile"
git push
```

## Experiment 1

Create a file called `first_mlfow_experiment.py` with the following contents:


```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Generate synthetic dataset
print("Generating data...")
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# Set experiment name (optional but recommended)
mlflow.set_experiment("our-experiment1")

# Start MLflow run
print("\nTraining model and logging with MLflow...")
with mlflow.start_run():
    # Define hyperparameters
    n_estimators = 100
    max_depth = 10
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Print results
    print(f"\n✅ Training completed!")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

```

Rebuild your docker image:

```bash
docker build -t ub-agile-data-science-mlflow .
```

Run our experiment with:

```bash
docker run -v .:/app ub-agile-data-science-mlflow:latest python first_mlfow_experiment.py
```

Check the results by running the mlflow UI and going to http://localhost:5000

```bash
docker run -p 5000:5000 -v .:/app ub-agile-data-science-mlflow:latest
```

Add the experiment file to git:

```
git add first_mlfow_experiment.py
git commit -m "Add initial experiment"
git push
```

## Experiment 2

Create a Python file `example_customer_churn_prediction.py` with the following contents:


```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def generate_customer_data(n_samples=1000):
    """Generate synthetic customer churn data."""
    np.random.seed(42)
    
    data = {
        'tenure': np.random.normal(24, 12, n_samples).clip(1, 72),
        'monthly_charges': np.random.normal(65, 20, n_samples).clip(20, 120),
        'total_charges': np.random.normal(1800, 800, n_samples).clip(50, 8000),
        'num_services': np.random.randint(0, 6, n_samples),
        'tech_support': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'contract_month_to_month': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    }
    
    df = pd.DataFrame(data)
    
    # Create target with logical relationships
    churn_prob = (
        0.1 +
        0.3 * df['contract_month_to_month'] +
        0.2 * (df['monthly_charges'] > 80) +
        0.15 * (df['tenure'] < 12) +
        0.1 * (df['tech_support'] == 0)
    ).clip(0, 0.9)
    
    df['churn'] = np.random.binomial(1, churn_prob, n_samples)
    
    return df

def train_and_log_model(model_type="random_forest", n_estimators=100, max_depth=10):
    """
    Train a model and log everything with MLflow.
    
    Args:
        model_type: "random_forest" or "logistic_regression"
        n_estimators: Number of trees (for random forest)
        max_depth: Max depth of trees (for random forest)
    """
    # Set experiment name
    mlflow.set_experiment("customer-churn-prediction")
    
    # Generate data
    print("Generating data...")
    df = generate_customer_data(n_samples=1000)
    
    # Prepare features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_type}_{n_estimators}_{max_depth}"):
        
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("churn_rate", y.mean())
        
        # Train model based on type
        if model_type == "random_forest":
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:  # logistic_regression
            mlflow.log_param("solver", "liblinear")
            mlflow.log_param("max_iter", 1000)
            
            # Scale features for logistic regression
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            model = LogisticRegression(
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
        
        print(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Create and log feature importance plot (for random forest)
        if model_type == "random_forest" and hasattr(model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title('Feature Importance')
            plt.tight_layout()
            
            # Save and log plot
            plt.savefig('feature_importance.png')
            mlflow.log_artifact('feature_importance.png')
            plt.close()
        
        # Log the model
        if model_type == "random_forest":
            mlflow.sklearn.log_model(model, "model")
        else:
            # For logistic regression, log both model and scaler
            mlflow.sklearn.log_model(model, "model")
            mlflow.sklearn.log_model(scaler, "scaler")
        
        # Log tags for organization
        mlflow.set_tag("team", "data-science")
        mlflow.set_tag("project", "customer-churn")
        mlflow.set_tag("model_family", model_type)
        
        print(f"\n✅ Run completed! View results in MLflow UI at http://localhost:5000")
        
        return metrics

def run_experiment_comparison():
    """Run multiple experiments with different configurations."""
    print("=" * 60)
    print("Running MLflow Experiment Comparison")
    print("=" * 60)
    
    experiments = [
        {"model_type": "random_forest", "n_estimators": 50, "max_depth": 5},
        {"model_type": "random_forest", "n_estimators": 100, "max_depth": 10},
        {"model_type": "random_forest", "n_estimators": 200, "max_depth": 15},
        {"model_type": "logistic_regression", "n_estimators": 0, "max_depth": 0},
    ]
    
    results = []
    for i, config in enumerate(experiments, 1):
        print(f"\n--- Experiment {i}/{len(experiments)} ---")
        metrics = train_and_log_model(**config)
        results.append({**config, **metrics})
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Experiment Comparison")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df[['model_type', 'n_estimators', 'accuracy', 'f1_score', 'roc_auc']])
    
    print("\n🎯 Best model by F1-score:")
    best_idx = results_df['f1_score'].idxmax()
    best_model = results_df.loc[best_idx]
    print(f"   {best_model['model_type']} with F1={best_model['f1_score']:.4f}")

if __name__ == "__main__":
    print("Running single MLflow experiment...\n")
    train_and_log_model(model_type="random_forest", n_estimators=100, max_depth=10)
```

Add `seaborn` to your `requirements.txt` file, it should look like this:

```
mlflow
seaborn
```

Rebuild your docker image:

```bash
docker build -t ub-agile-data-science-mlflow .
```

Run our experiment with:

```bash
docker run -v .:/app ub-agile-data-science-mlflow:latest python example_customer_churn_prediction.py
```

Check the results by running the mlflow UI and going to http://localhost:5000

```bash
docker run -p 5000:5000 -v .:/app ub-agile-data-science-mlflow:latest
```

Check the new generated graph stored as artifact.

Add the experiment file to git:

```
git add first_mlfow_experiment.py requirements.txt
git commit -m "Add example_customer_churn_prediction.py"
git push
```

### Experiment 2b

Edit the file `example_customer_churn_prediction.py` and update the `if __name__` section with:

```python
if __name__ == "__main__":
    run_experiment_comparison()
```

Rebuild your docker image:

```bash
docker build -t ub-agile-data-science-mlflow .
```

Run our experiment with:

```bash
docker run -v .:/app ub-agile-data-science-mlflow:latest python example_customer_churn_prediction.py
```

Check the results by running the mlflow UI and going to http://localhost:5000

```bash
docker run -p 5000:5000 -v .:/app ub-agile-data-science-mlflow:latest
```

Add the updated file to git:

```
git add first_mlfow_experiment.py
git commit -m "Update example_customer_churn_prediction.py to run more than one experiment"
git push
```
