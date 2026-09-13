import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/features_chennai.csv"

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

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

df["forecast_issue_time"] = pd.to_datetime(
    df["forecast_issue_time"]
)

df = df.sort_values(
    ["forecast_issue_time", "target_time"]
).reset_index(drop=True)

init_dates = sorted(
    df["forecast_issue_time"].dt.date.unique()
)

print(f"Total cases: {len(df)}")
print(f"Initialization dates: {len(init_dates)}")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n_dates = len(init_dates)

train_end = int(n_dates * 0.60)
val_end = int(n_dates * 0.80)

train_dates = init_dates[:train_end]
val_dates = init_dates[train_end:val_end]
test_dates = init_dates[val_end:]

train_df = df[
    df["forecast_issue_time"].dt.date.isin(train_dates)
].copy()

val_df = df[
    df["forecast_issue_time"].dt.date.isin(val_dates)
].copy()

test_df = df[
    df["forecast_issue_time"].dt.date.isin(test_dates)
].copy()


print("\n" + "=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

print("\nTraining dates:")
print(train_dates)

print("\nValidation dates:")
print(val_dates)

print("\nTest dates:")
print(test_dates)

print(
    f"\nCases: "
    f"train={len(train_df)}, "
    f"validation={len(val_df)}, "
    f"test={len(test_df)}"
)


# ============================================================
# CREATE LEAKAGE-FREE LABEL THRESHOLD
# ============================================================

# The bust definition depends on forecast error.
# Therefore the error threshold must be calculated ONLY
# from the training period.

error_threshold = train_df[
    "absolute_control_error"
].quantile(0.90)

print("\n" + "=" * 80)
print("LABEL THRESHOLD")
print("=" * 80)

print(
    f"Training-only 90th percentile error: "
    f"{error_threshold:.4f} mm"
)


def create_labels(data):

    event_significant = (
        (data["observed_rainfall_mm"] >= 1.0)
        |
        (data["control_forecast"] >= 1.0)
    )

    bust = (
        (data["absolute_control_error"] >= error_threshold)
        &
        event_significant
    )

    return bust.astype(int)


train_df["target"] = create_labels(train_df)
val_df["target"] = create_labels(val_df)
test_df["target"] = create_labels(test_df)


# ============================================================
# TRAIN MODEL
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df["target"]

X_val = val_df[FEATURES]
y_val = val_df["target"]

X_test = test_df[FEATURES]
y_test = test_df["target"]


positive = y_train.sum()
negative = len(y_train) - positive

scale_pos_weight = (
    negative / positive
    if positive > 0
    else 1
)

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

print(f"Training busts: {positive}")
print(f"Training normal: {negative}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


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


model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]


print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)


# Choose threshold ONLY using validation data.
thresholds = np.round(
    np.arange(0.50, 1.00, 0.01),
    2
)

results = []

for threshold in thresholds:

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


threshold_df = pd.DataFrame(results)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

print(
    f"\nBest validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation precision: "
    f"{best_row['precision']:.3f}"
)

print(
    f"Validation recall: "
    f"{best_row['recall']:.3f}"
)

print(
    f"Validation F1: "
    f"{best_row['f1']:.3f}"
)


# ============================================================
# FINAL TEST
# ============================================================

test_prob = model.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)


precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_prob
)

pr_auc = average_precision_score(
    y_test,
    test_prob
)

cm = confusion_matrix(
    y_test,
    test_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL UNSEEN TEST RESULTS")
print("=" * 80)

print(f"Threshold : {best_threshold:.2f}")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1        : {f1:.3f}")
print(f"ROC-AUC   : {roc_auc:.3f}")
print(f"PR-AUC    : {pr_auc:.3f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTest busts:", int(y_test.sum()))
print("Test normal:", int((y_test == 0).sum()))


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

output = test_df[
    [
        "forecast_issue_time",
        "target_time",
        "control_forecast",
        "ensemble_mean",
        "ensemble_spread",
        "observed_rainfall_mm",
        "bust_type",
    ]
].copy()

output["actual_bust"] = y_test.values
output["bust_probability"] = test_prob
output["predicted_bust"] = test_pred

output.to_csv(
    "data/final_test_predictions.csv",
    index=False
)

print(
    "\nSaved test predictions to "
    "data/final_test_predictions.csv"
)