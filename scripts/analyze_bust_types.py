import pandas as pd


INPUT_FILE = "data/final_test_predictions.csv"
OUTPUT_FILE = "data/bust_type_analysis.csv"


print("Loading final test predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Cases: {len(df)}")


# ============================================================
# BUST TYPE CLASSIFICATION
# ============================================================

def classify_bust_type(row):
    """
    Classify the reason for a detected forecast bust.

    This diagnostic is applied only to cases that the model
    predicts as busts.
    """

    if row["predicted_bust"] != 1:
        return "NORMAL"

    forecast = row["control_forecast"]
    observed = row["observed_rainfall_mm"]

    # Forecast missed a significant rainfall event
    if forecast < 1.0 and observed >= 1.0:
        return "MISSED EVENT"

    # Forecast predicted a significant event that did not occur
    if forecast >= 1.0 and observed < 1.0:
        return "FALSE ALARM"

    # Both forecast and observation indicate an event.
    # Since the model has already identified this as a bust,
    # classify it as a magnitude error.
    if forecast >= 1.0 and observed >= 1.0:
        return "MAGNITUDE ERROR"

    # Remaining detected busts cannot be reliably assigned
    # to one of the three categories from these thresholds.
    return "UNCLASSIFIED BUST"


df["detected_bust_type"] = df.apply(
    classify_bust_type,
    axis=1
)


# ============================================================
# SUMMARY OF DETECTED BUSTS
# ============================================================

print("\n" + "=" * 80)
print("DETECTED BUST TYPES")
print("=" * 80)

detected_busts = df[
    df["predicted_bust"] == 1
]

print(
    detected_busts["detected_bust_type"]
    .value_counts()
)


# ============================================================
# ACTUAL BUST TYPES
# ============================================================

def classify_actual_type(row):

    if row["actual_bust"] != 1:
        return "NORMAL"

    forecast = row["control_forecast"]
    observed = row["observed_rainfall_mm"]

    if forecast < 1.0 and observed >= 1.0:
        return "MISSED EVENT"

    if forecast >= 1.0 and observed < 1.0:
        return "FALSE ALARM"

    if forecast >= 1.0 and observed >= 1.0:
        return "MAGNITUDE ERROR"

    return "UNCLASSIFIED BUST"


df["actual_bust_type"] = df.apply(
    classify_actual_type,
    axis=1
)


print("\n" + "=" * 80)
print("ACTUAL BUST TYPES")
print("=" * 80)

actual_busts = df[
    df["actual_bust"] == 1
]

print(
    actual_busts["actual_bust_type"]
    .value_counts()
)


# ============================================================
# TYPE DETECTION PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("BUST TYPE DETECTION")
print("=" * 80)

for bust_type in [
    "MISSED EVENT",
    "FALSE ALARM",
    "MAGNITUDE ERROR"
]:

    actual_count = (
        actual_busts["actual_bust_type"] == bust_type
    ).sum()

    detected_count = (
        detected_busts["detected_bust_type"] == bust_type
    ).sum()

    correct_count = (
        (
            df["detected_bust_type"] == bust_type
        )
        &
        (
            df["actual_bust_type"] == bust_type
        )
    ).sum()

    detection_rate = (
        correct_count / actual_count
        if actual_count > 0
        else 0
    )

    print(f"\n{bust_type}")
    print(f"Actual:          {actual_count}")
    print(f"Detected:        {detected_count}")
    print(f"Correct:         {correct_count}")
    print(f"Detection rate:  {detection_rate:.1%}")


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)


# ============================================================
# CLEAN DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("TOP DETECTED BUSTS")
print("=" * 80)

display_columns = [
    "target_time",
    "bust_probability",
    "control_forecast",
    "ensemble_mean",
    "ensemble_spread",
    "detected_bust_type",
    "actual_bust",
]

top_cases = (
    detected_busts
    .sort_values(
        "bust_probability",
        ascending=False
    )
    .head(20)
    .copy()
)

# Explicit numeric formatting prevents values from visually
# running together in the terminal.

top_cases["bust_probability"] = (
    top_cases["bust_probability"]
    .map(lambda x: f"{x:.3f}")
)

top_cases["control_forecast"] = (
    top_cases["control_forecast"]
    .map(lambda x: f"{x:.2f}")
)

top_cases["ensemble_mean"] = (
    top_cases["ensemble_mean"]
    .map(lambda x: f"{x:.2f}")
)

top_cases["ensemble_spread"] = (
    top_cases["ensemble_spread"]
    .map(lambda x: f"{x:.2f}")
)

print(
    top_cases[
        display_columns
    ].to_string(index=False)
)