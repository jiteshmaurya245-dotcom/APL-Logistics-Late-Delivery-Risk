# ============================================================
# APL LOGISTICS — Standalone model training script
# Run this once (or whenever the dataset changes) to produce a
# cached model, instead of relying on the Streamlit app to
# retrain from scratch on every cold start.
#
# IMPORTANT — model size is deliberately constrained (max_depth,
# min_samples_leaf). An earlier unconstrained RandomForest grew
# to a 650MB pickle (full-purity leaves on 144K training rows),
# which exceeded Streamlit Community Cloud's ~1GB memory limit
# during training and crashed the deployed app with a silent
# OOM kill (no Python traceback — just "Oh no."). This config
# produces a ~10MB model. A recalibrated 0.35 decision threshold
# (instead of the default 0.5) actually gives BETTER recall than
# the original oversized model, since recall matters more than
# precision for this use case (missing a true delay costs more
# than an unnecessary check-in on a false alarm).
#
# HOW TO RUN (from the repo root):
#   pip install -r requirements.txt
#   python scripts/train_model.py
#
# Produces:
#   data/model.pkl        - trained RandomForestClassifier
#   data/scaler.pkl        - fitted StandardScaler
#   data/feature_cols.json - training column order (needed to
#                            align single-order predictions)
#   data/metrics.json      - AUC / F1 / precision / recall
# ============================================================

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "APL_Logistics.csv"

DECISION_THRESHOLD = 0.35  # see rationale in header comment above

RF_PARAMS = dict(
    n_estimators=100, max_depth=12, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)

DROP_COLS = [
    "Customer Fname", "Customer Lname", "Customer Street", "Customer Zipcode",
    "Customer City", "Customer State", "Customer Country", "Order City",
    "Order Country", "Order State", "Customer Id", "Order Customer Id",
    "Category Id", "Department Id", "Latitude", "Longitude", "Product Name",
    "Delivery Status", "Days for shipping (real)",
    # ^ dropped deliberately: both are direct proxies for the target and
    # would leak the answer into the model (see project README).
]


def train():
    df = pd.read_csv(CSV_PATH, encoding="latin1")
    df_raw = df.drop(columns=DROP_COLS).dropna().reset_index(drop=True)

    df_ml = df_raw.copy()
    df_ml["shipping_pressure"] = df_ml["Days for shipment (scheduled)"] / (df_ml["Order Item Quantity"] + 1)
    df_ml["discount_flag"] = (df_ml["Order Item Discount"] > 0).astype(int)
    df_ml["profit_flag"] = (df_ml["Order Profit Per Order"] < 0).astype(int)
    df_ml["high_discount"] = (df_ml["Order Item Discount Rate"] > 0.15).astype(int)
    df_ml["low_scheduled_days"] = (df_ml["Days for shipment (scheduled)"] <= 2).astype(int)

    df_enc = pd.get_dummies(
        df_ml, columns=df_ml.select_dtypes(include=["object", "string"]).columns.tolist(),
        drop_first=True, dtype=int,
    )
    X = df_enc.drop(columns=["Late_delivery_risk"])
    y = df_enc["Late_delivery_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    scaler = StandardScaler()
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train_s[num_cols])
    X_test_s[num_cols] = scaler.transform(X_test_s[num_cols])

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)
    metrics = {
        "auc": round(roc_auc_score(y_test, probs), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "decision_threshold": DECISION_THRESHOLD,
        "n_orders": len(df_raw),
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(DATA_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(DATA_DIR / "feature_cols.json", "w") as f:
        json.dump({"feature_cols": X_train_s.columns.tolist(), "num_cols": num_cols}, f)
    with open(DATA_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    model_mb = (DATA_DIR / "model.pkl").stat().st_size / 1e6
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"model.pkl size: {model_mb:.1f} MB", "(WARNING: large — check RF_PARAMS)" if model_mb > 50 else "(OK for git + cloud deployment)")


if __name__ == "__main__":
    train()
