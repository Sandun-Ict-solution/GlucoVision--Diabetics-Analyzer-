# model_training.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Set the correct dataset path
FILE_PATH = r"D:\sandun\Machine_learnig_Model\data\dataset.csv"

def load_data(file_path):
    """Load dataset and validate its existence."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset from {file_path}")
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")

def validate_columns(df, required_columns, target_column):
    """Validate that required columns exist in the dataset."""
    missing_cols = [col for col in required_columns + [target_column] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

def preprocess_data(df, zero_as_missing):
    """Preprocess data by handling missing values."""
    validate_columns(df, zero_as_missing, "Outcome")
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    imputer = SimpleImputer(strategy="median")
    df[zero_as_missing] = imputer.fit_transform(df[zero_as_missing])
    return df

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }
    return metrics

def print_metrics(name, metrics):
    """Print evaluation metrics in a formatted way."""
    print(f"\n=== {name} Test Metrics ===")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"{k}:\n{v}")
        elif k == "classification_report":
            print(f"{k}:\n{v}")
        else:
            print(f"{k}: {v:.4f}")

def main():
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    scoring_metric = "roc_auc"
    
    try:
        # 1. Load dataset
        df = load_data(FILE_PATH)

        # 2. Quick EDA
        print("\nDataset Info:")
        print("Shape:", df.shape)
        print("\nHead:\n", df.head())
        print("\nInfo:")
        print(df.info())
        print("\nDescribe:\n", df.describe())

        # 3. Preprocess data
        df = preprocess_data(df, zero_as_missing)

        # 4. Features and target
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # 5. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 6. Pipelines for scaling + model
        scaler = StandardScaler()
        log_pipe = Pipeline([
            ("scaler", scaler),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        rf_pipe = Pipeline([
            ("scaler", scaler),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        # 7. Cross-validation comparison
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        models = {"LogisticRegression": log_pipe, "RandomForest": rf_pipe}
        
        print("\nCross-Validation Results:")
        for name, pipe in models.items():
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring_metric)
            print(f"{name} CV {scoring_metric.upper()}: mean={scores.mean():.4f} std={scores.std():.4f}")

        # 8. Fit and evaluate
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            results[name] = evaluate_model(model, X_test, y_test)
            print_metrics(name, results[name])

        # 9. Select best model
        best_name = max(results, key=lambda n: results[n]["roc_auc"])
        best_model = models[best_name]
        print(f"\nSelected best model based on ROC-AUC: {best_name}")

        # 10. Save model
        joblib.dump(best_model, "model.pkl")
        print("Saved model to model.pkl")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
