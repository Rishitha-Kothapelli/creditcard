"""
main.py
--------
FastAPI backend for the Credit Card Fraud Detection dashboard.

Run with:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload

Then open:
    http://127.0.0.1:8000
"""

import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

MODEL_PATH = os.path.join("model", "fraud_model.pkl")
TEST_DATA_PATH = os.path.join("model", "test_data.pkl")
STATIC_DIR = "static"

MIN_THRESHOLD = 0.10
MAX_THRESHOLD = 0.90

app = FastAPI(title="Credit Card Fraud Detection API")

# ------------------------------------------------------------------
# Load the trained model and pre-computed test probabilities ONCE at
# startup. The threshold is applied on-the-fly on these stored
# probabilities, so the model is never retrained per request.
# ------------------------------------------------------------------
model_bundle = None
y_test = None
fraud_probabilities = None
load_error = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' not found. Run 'python train_model.py' first."
        )
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(
            f"'{TEST_DATA_PATH}' not found. Run 'python train_model.py' first."
        )

    model_bundle = joblib.load(MODEL_PATH)
    test_data = joblib.load(TEST_DATA_PATH)
    y_test = np.asarray(test_data["y_test"])
    fraud_probabilities = np.asarray(test_data["fraud_probabilities"])
except Exception as e:
    # Keep the server up so the frontend can show a clear error message,
    # instead of crashing on startup.
    load_error = str(e)
    print(f"WARNING: {load_error}")


# ------------------------------------------------------------------
# Serve static frontend files (style.css, script.js, etc.)
# ------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    """Serve the dashboard HTML page at http://127.0.0.1:8000/"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="index.html not found in static/")
    return FileResponse(index_path)


@app.get("/api/metrics")
def get_metrics(threshold: float = Query(0.50, description="Fraud classification threshold")):
    """
    Apply the given threshold to the pre-computed fraud probabilities
    (from the test set) and return accuracy, precision, recall, and the
    confusion matrix as JSON. Does NOT retrain the model.
    """
    if load_error:
        raise HTTPException(
            status_code=500,
            detail=f"Model/test data not available: {load_error}",
        )

    # Validate threshold
    if threshold < MIN_THRESHOLD or threshold > MAX_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}.",
        )

    try:
        # Apply threshold manually to stored probabilities (no retraining)
        predictions = (fraud_probabilities >= threshold).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return JSONResponse(
            {
                "threshold": round(threshold, 2),
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {e}")


@app.get("/api/health")
def health_check():
    """Simple health check, also reports whether the model loaded correctly."""
    return {
        "status": "ok" if not load_error else "error",
        "detail": load_error,
        "test_samples": int(len(y_test)) if y_test is not None else 0,
    }
