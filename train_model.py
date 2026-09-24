"""
train_model.py
---------------
Trains a Logistic Regression fraud-detection model ONCE on creditcard.csv
and saves:
    model/fraud_model.pkl   -> the trained LogisticRegression model
    model/test_data.pkl     -> dict with y_test and fraud probabilities
                                for the test set (used by the API so it
                                never has to retrain when the threshold
                                changes on the frontend)

Run this once before starting the FastAPI server:
    python train_model.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "creditcard.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
TEST_DATA_PATH = os.path.join(MODEL_DIR, "test_data.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    # 1. Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: '{DATA_PATH}' not found in the project root.")
        print("Download the dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("and place it as 'creditcard.csv' here, OR run:")
        print("  python generate_sample_data.py")
        print("to create a synthetic dataset for testing.")
        sys.exit(1)

    print(f"Loading dataset from '{DATA_PATH}' ...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"ERROR: Failed to read '{DATA_PATH}': {e}")
        sys.exit(1)

    if "Class" not in df.columns:
        print("ERROR: Dataset must contain a 'Class' column (0 = genuine, 1 = fraud).")
        sys.exit(1)

    if df.isnull().values.any():
        print("Warning: dataset contains missing values. Dropping rows with NaNs.")
        df = df.dropna()

    # 2. Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"Dataset loaded: {len(df)} rows, "
          f"{int(y.sum())} fraud / {int((y == 0).sum())} genuine.")

    # 3. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale features (helps Logistic Regression converge and perform better,
    # especially on 'Amount' and 'Time' which are on very different scales).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Handle class imbalance + 5. Train the model
    print("Training Logistic Regression model (class_weight='balanced') ...")
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 8. Use predict_proba() to obtain fraud probabilities on the test set
    fraud_probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # 6. Save the trained model (and the scaler, needed to transform future data
    #    consistently) using Joblib.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)

    # 7. Save the test data / probabilities needed to compute metrics later,
    #    so the API never has to retrain when the threshold changes.
    joblib.dump(
        {
            "y_test": y_test.to_numpy(),
            "fraud_probabilities": fraud_probabilities,
        },
        TEST_DATA_PATH,
    )

    print(f"Model saved to '{MODEL_PATH}'")
    print(f"Test data saved to '{TEST_DATA_PATH}'")
    print("Training complete. You can now start the FastAPI server:")
    print("  uvicorn main:app --host 127.0.0.1 --port 8000 --reload")


if __name__ == "__main__":
    main()
