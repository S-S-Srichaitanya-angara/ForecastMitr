import pandas as pd
import numpy as np

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "features_chennai.csv"


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


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "forecast_issue_time",
        "target_time",
    ],
)


print("=" * 70)
print("FORECASTMITR — XGBOOST BUST DETECTION")
print("=" * 70)

print(f"Total cases: {len(df)}")


# ---------------------------------------------------------
# SPLIT BY INITIALIZATION DATE
# ---------------------------------------------------------
#
# IMPORTANT:
# Never split individual rows randomly.
# All forecast times from one initialization must stay
# together.
#

initializations = sorted(
    df["forecast_issue_time"].unique()
)

print(f"Initialization dates: {len(initializations)}")

split_index = int(len(initializations) * 0.70)

train_initializations = initializations[:split_index]
test_initializations = initializations[split_index:]


train = df[
    df["forecast_issue_time"].isin(train_initializations)
].copy()

test = df[
    df["forecast_issue_time"].isin(test_initializations)
].copy()


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(f"Training initializations: {len(train_initializations)}")
print(f"Testing initializations : {len(test_initializations)}")

print(f"Training cases: {len(train)}")
print(f"Testing cases : {len(test)}")

print("\nTraining dates:")
for date in train_initializations:
    print(pd.Timestamp(date))

print("\nTesting dates:")
for date in test_initializations:
    print(pd.Timestamp(date))


# ---------------------------------------------------------
# FEATURES / TARGET
# ---------------------------------------------------------

X_train = train[FEATURES]
y_train = train["bust"]

X_test = test[FEATURES]
y_test = test["bust"]


print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    "Training:",
    y_train.value_counts().sort_index().to_dict()
)

print(
    "Testing :",
    y_test.value_counts().sort_index().to_dict()
)


# ---------------------------------------------------------
# CLASS WEIGHT
# ---------------------------------------------------------

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
    if positive_count > 0
    else 1.0
)

print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")


# ---------------------------------------------------------
# XGBOOST MODEL
# ---------------------------------------------------------

model = XGBClassifier(
    n_estimators=250,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    reg_alpha=0.2,
    reg_lambda=2.0,

    random_state=42,

    n_jobs=-1,
)


print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

model.fit(
    X_train,
    y_train,
)


print("Training complete.")


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


test = test.copy()

test["bust_probability"] = probabilities
test["predicted_bust"] = predictions


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)


print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ROC-AUC requires both classes in test
if y_test.nunique() == 2:

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

else:

    print("ROC-AUC   : unavailable")
    print("PR-AUC    : unavailable")


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    predictions,
)

print(cm)

print(
    "\nRows = actual"
    "\nColumns = predicted"
)

print(
    "              Predicted Normal   Predicted Bust"
)

print(
    f"Actual Normal      {cm[0,0]:4d}             {cm[0,1]:4d}"
)

print(
    f"Actual Bust        {cm[1,0]:4d}             {cm[1,1]:4d}"
)


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Bust",
        ],
        zero_division=0,
    )
)


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
})

importance = importance.sort_values(
    "importance",
    ascending=False,
)

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------
# MOST CONFIDENT PREDICTIONS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("HIGHEST BUST-RISK CASES")
print("=" * 70)

columns = [
    "forecast_issue_time",
    "target_time",
    "forecast_time_hours",
    "control_forecast",
    "ensemble_mean",
    "ensemble_spread",
    "bust",
    "bust_probability",
]


high_risk = test.sort_values(
    "bust_probability",
    ascending=False,
).head(15)


print(
    high_risk[columns].to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE MODEL PREDICTIONS
# ---------------------------------------------------------

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "xgboost_predictions.csv"
)

test.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Predictions saved to:\n{OUTPUT_FILE}"
)