
"""
Complete MLflow Example - Customer Churn Prediction
This script demonstrates MLflow tracking with a realistic example.
"""

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
    run_experiment_comparison()
