"""
generate_sample_data.py
------------------------
OPTIONAL helper script.

The real "Credit Card Fraud Detection" dataset from Kaggle is ~150 MB and is
NOT included here (and can't be auto-downloaded from this environment).
Download it yourself from:

    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

and place the file as `creditcard.csv` in this project's root folder.

If you just want to test-drive the app immediately without downloading
anything, run this script once to generate a synthetic `creditcard.csv`
with the same column structure (Time, V1..V28, Amount, Class) and a
realistic ~0.17% fraud rate. It is for demo purposes only — swap in the
real Kaggle CSV for a genuine project.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 20000
FRAUD_RATE = 0.0017  # similar imbalance to the real dataset

rng = np.random.default_rng(RANDOM_STATE)

n_fraud = max(1, int(N_ROWS * FRAUD_RATE))
n_genuine = N_ROWS - n_fraud

# Genuine transactions: V1-V28 centered near 0 with small spread
genuine_v = rng.normal(loc=0.0, scale=1.0, size=(n_genuine, 28))
# Fraudulent transactions: shifted / more spread out, so a model can learn a signal
fraud_v = rng.normal(loc=2.5, scale=2.5, size=(n_fraud, 28))

genuine_amount = np.round(np.abs(rng.normal(loc=60, scale=50, size=n_genuine)), 2)
fraud_amount = np.round(np.abs(rng.normal(loc=120, scale=100, size=n_fraud)), 2)

time_col = np.sort(rng.integers(low=0, high=172792, size=N_ROWS))

V = np.vstack([genuine_v, fraud_v])
amount = np.concatenate([genuine_amount, fraud_amount])
label = np.concatenate([np.zeros(n_genuine, dtype=int), np.ones(n_fraud, dtype=int)])

df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
df.insert(0, "Time", time_col)
df["Amount"] = amount
df["Class"] = label

# Shuffle rows so fraud cases aren't all at the bottom
df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

df.to_csv("creditcard.csv", index=False)
print(f"Synthetic creditcard.csv generated with {N_ROWS} rows "
      f"({n_fraud} fraud / {n_genuine} genuine).")
print("Replace this file with the real Kaggle dataset for an actual project.")
