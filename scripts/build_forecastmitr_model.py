import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/features_chennai.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "forecastmitr_xgboost.joblib"
CONFIG_FILE = MODEL_DIR / "forecastmitr_config.json"

FEATURES = [
    "control_forecast",
    "ensemble_mean",
    "ensemble_median",
    "ensemble_min",
    "ensemble_max",
    "ensemble_spread",
    "forecast_time_hours",
    "month",
    "control_vs_ensemble",
    "relative_spread",
]

# Chronological split
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# Selected using the validation set
DECISION_THRESHOLD = 0.69

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

df["forecast_issue_time"] = pd.to_datetime(df["forecast_issue_time"])
df["target_time"] = pd.to_datetime(df["target_time"])

init_dates = sorted(df["forecast_issue_time"].dt.date.unique())

n_dates = len(init_dates)

train_end = int(n_dates * TRAIN_RATIO)
validation_end = int(n_dates * (TRAIN_RATIO + VALIDATION_RATIO))

train_dates = init_dates[:train_end]
validation_dates = init_dates[train_end:validation_end]
test_dates = init_dates[validation_end:]

train_df = df[df["forecast_issue_time"].dt.date.isin(train_dates)].copy()
validation_df = df[
    df["forecast_issue_time"].dt.date.isin(validation_dates)
].copy()
test_df = df[df["forecast_issue_time"].dt.date.isin(test_dates)].copy()


# ============================================================
# CREATE LEAKAGE-FREE TRAINING LABEL
# ============================================================

print("\nCreating training labels...")

# IMPORTANT:
# Bust threshold is calculated ONLY from training data.

error_threshold = train_df["absolute_control_error"].quantile(0.90)

train_df["event_significant"] = (
    (train_df["observed_rainfall_mm"] >= 1.0)
    | (train_df["control_forecast"] >= 1.0)
)

train_df["training_bust"] = (
    (train_df["absolute_control_error"] >= error_threshold)
    & train_df["event_significant"]
).astype(int)


# ============================================================
# TRAIN MODEL
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df["training_bust"]

positive_count = int(y_train.sum())
negative_count = int((y_train == 0).sum())

scale_pos_weight = negative_count / positive_count

print("\nTraining information")
print("=" * 60)
print(f"Training dates       : {len(train_dates)}")
print(f"Training cases       : {len(train_df)}")
print(f"Training busts       : {positive_count}")
print(f"Training normal      : {negative_count}")
print(f"Error threshold      : {error_threshold:.4f} mm")
print(f"Scale positive weight: {scale_pos_weight:.2f}")


model = XGBClassifier(
    n_estimators=250,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.2,
    reg_lambda=2,
    random_state=RANDOM_STATE,
)

model.fit(X_train, y_train)


# ============================================================
# VALIDATION CHECK
# ============================================================

from sklearn.metrics import precision_score, recall_score, f1_score

X_val = validation_df[FEATURES]

# Validation labels are generated using the TRAINING threshold.
validation_df["event_significant"] = (
    (validation_df["observed_rainfall_mm"] >= 1.0)
    | (validation_df["control_forecast"] >= 1.0)
)

validation_df["validation_bust"] = (
    (validation_df["absolute_control_error"] >= error_threshold)
    & validation_df["event_significant"]
).astype(int)

val_probability = model.predict_proba(X_val)[:, 1]

val_prediction = (
    val_probability >= DECISION_THRESHOLD
).astype(int)

val_precision = precision_score(
    validation_df["validation_bust"],
    val_prediction,
    zero_division=0,
)

val_recall = recall_score(
    validation_df["validation_bust"],
    val_prediction,
    zero_division=0,
)

val_f1 = f1_score(
    validation_df["validation_bust"],
    val_prediction,
    zero_division=0,
)

print("\nValidation performance")
print("=" * 60)
print(f"Decision threshold: {DECISION_THRESHOLD:.2f}")
print(f"Precision         : {val_precision:.3f}")
print(f"Recall            : {val_recall:.3f}")
print(f"F1                : {val_f1:.3f}")


# ============================================================
# SAVE MODEL + CONFIGURATION
# ============================================================

joblib.dump(model, MODEL_FILE)

config = {
    "model_type": "XGBoost",
    "features": FEATURES,
    "decision_threshold": DECISION_THRESHOLD,
    "training_error_threshold_mm": float(error_threshold),
    "event_rainfall_threshold_mm": 1.0,
    "train_dates": [str(x) for x in train_dates],
    "validation_dates": [str(x) for x in validation_dates],
    "test_dates": [str(x) for x in test_dates],
    "random_state": RANDOM_STATE,
}

with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=4)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL BUILD COMPLETE")
print("=" * 60)

print(f"Model saved : {MODEL_FILE}")
print(f"Config saved: {CONFIG_FILE}")
print("\nForecastMitr model is ready for inference.")