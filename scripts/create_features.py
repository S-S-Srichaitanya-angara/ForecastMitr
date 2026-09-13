import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "bust_dataset_chennai.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "features_chennai.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "forecast_issue_time",
        "target_time",
    ],
)

print("=" * 70)
print("CREATING FEATURES FOR WEATHER BUST DETECTION")
print("=" * 70)

print(f"Input rows: {len(df)}")


# ------------------------------------------------------------
# TIME FEATURES
# ------------------------------------------------------------

df["month"] = df["target_time"].dt.month

df["season"] = (
    df["month"]
    .map({
        1: "winter",
        2: "winter",
        3: "summer",
        4: "summer",
        5: "summer",
        6: "monsoon",
        7: "monsoon",
        8: "monsoon",
        9: "monsoon",
        10: "post_monsoon",
        11: "post_monsoon",
        12: "winter",
    })
)


# ------------------------------------------------------------
# ENSEMBLE FEATURES
# ------------------------------------------------------------

df["control_vs_ensemble"] = (
    df["control_forecast"]
    - df["ensemble_mean"]
)


df["relative_spread"] = (
    df["ensemble_spread"]
    / (df["ensemble_mean"] + 0.1)
)


df["ensemble_range_ratio"] = (
    df["ensemble_spread"]
    / (df["ensemble_mean"] + 0.1)
)


# ------------------------------------------------------------
# FORECAST INTENSITY
# ------------------------------------------------------------

df["forecast_intensity"] = (
    df["ensemble_mean"]
)


# ------------------------------------------------------------
# SELECT MODEL FEATURES
# ------------------------------------------------------------

feature_columns = [
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


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

print("\nFeatures:")

for feature in feature_columns:
    print(
        f"{feature:<25} "
        f"missing={df[feature].isna().sum():<5} "
        f"min={df[feature].min():.4f} "
        f"max={df[feature].max():.4f}"
    )


print("\nTarget distribution:")

print(
    df["bust"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# LEAKAGE CHECK
# ------------------------------------------------------------

forbidden_columns = [
    "observed_rainfall_mm",
    "control_error",
    "absolute_control_error",
    "ensemble_mean_error",
    "absolute_ensemble_mean_error",
]


print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

leakage_found = []

for column in forbidden_columns:
    if column in feature_columns:
        leakage_found.append(column)


if leakage_found:
    print("WARNING: Leakage detected!")
    print(leakage_found)
else:
    print("No target leakage in model features.")


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Rows:   {len(df)}")
print(f"Features: {len(feature_columns)}")
print(f"Output: {OUTPUT_FILE}")

print("\nModel features:")

for feature in feature_columns:
    print(" -", feature)