import pandas as pd
import numpy as np


INPUT_FILE = "data/final_test_predictions.csv"
OUTPUT_FILE = "data/final_risk_predictions.csv"


# ============================================================
# LOAD
# ============================================================

print("Loading predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Cases: {len(df)}")


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    if probability >= 0.85:
        return "CRITICAL"

    elif probability >= 0.70:
        return "HIGH"

    elif probability >= 0.40:
        return "MODERATE"

    else:
        return "LOW"


df["risk_level"] = df["bust_probability"].apply(
    get_risk_level
)


# ============================================================
# ENSEMBLE DISAGREEMENT
# ============================================================

df["ensemble_disagreement"] = (
    df["ensemble_spread"]
)


# ============================================================
# PRIMARY SIGNAL
# ============================================================

def determine_signal(row):

    control = row["control_forecast"]
    ensemble_mean = row["ensemble_mean"]
    spread = row["ensemble_spread"]

    # Strong control forecast
    if control >= 3.0:
        return "High control forecast intensity"

    # Large ensemble disagreement
    if spread >= 5.0:
        return "Large ensemble disagreement"

    # Control substantially exceeds ensemble
    if control > ensemble_mean * 2 and control >= 1.0:
        return "Control forecast differs strongly from ensemble"

    # Ensemble substantially exceeds control
    if ensemble_mean > control * 2 and ensemble_mean >= 1.0:
        return "Ensemble indicates higher precipitation than control"

    # Moderate spread
    if spread >= 2.0:
        return "Moderate ensemble disagreement"

    return "Weak bust signal"


df["primary_signal"] = df.apply(
    determine_signal,
    axis=1
)


# ============================================================
# HUMAN-READABLE EXPLANATION
# ============================================================

def generate_explanation(row):

    probability = row["bust_probability"]
    risk = row["risk_level"]

    control = row["control_forecast"]
    ensemble = row["ensemble_mean"]
    spread = row["ensemble_spread"]

    signal = row["primary_signal"]

    if risk == "CRITICAL":

        prefix = (
            "Very high probability of forecast bust."
        )

    elif risk == "HIGH":

        prefix = (
            "High probability of forecast bust."
        )

    elif risk == "MODERATE":

        prefix = (
            "Moderate probability of forecast bust."
        )

    else:

        prefix = (
            "Low probability of forecast bust."
        )

    return (
        f"{prefix} "
        f"Model probability: {probability:.0%}. "
        f"Control forecast: {control:.2f} mm/3h. "
        f"Ensemble mean: {ensemble:.2f} mm/3h. "
        f"Ensemble spread: {spread:.2f} mm. "
        f"Primary signal: {signal}."
    )


df["explanation"] = df.apply(
    generate_explanation,
    axis=1
)


# ============================================================
# ALERT
# ============================================================

def create_alert(row):

    if row["risk_level"] == "CRITICAL":
        return "RED ALERT"

    elif row["risk_level"] == "HIGH":
        return "HIGH RISK"

    elif row["risk_level"] == "MODERATE":
        return "WATCH"

    return "NORMAL"


df["alert"] = df.apply(
    create_alert,
    axis=1
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("RISK SCORE SUMMARY")
print("=" * 80)

print(
    df["risk_level"]
    .value_counts()
    .reindex(
        ["CRITICAL", "HIGH", "MODERATE", "LOW"],
        fill_value=0
    )
)

print("\nAlerts:")
print(
    df["alert"]
    .value_counts()
)

print("\nTop 10 highest-risk cases:")

columns = [
    "target_time",
    "bust_probability",
    "risk_level",
    "control_forecast",
    "ensemble_mean",
    "ensemble_spread",
    "primary_signal",
    "actual_bust",
    "predicted_bust",
]

print(
    df.sort_values(
        "bust_probability",
        ascending=False
    )[columns].head(10).to_string(index=False)
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)