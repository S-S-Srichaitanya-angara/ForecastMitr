import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "forecast_observation_chennai.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "bust_dataset_chennai.csv"
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
print("CREATING WEATHER BUST LABELS")
print("=" * 70)

print(f"Input cases: {len(df)}")


# ------------------------------------------------------------
# ERROR THRESHOLD
# ------------------------------------------------------------

error_threshold = df[
    "absolute_control_error"
].quantile(0.90)


print(
    f"\n90th percentile error: "
    f"{error_threshold:.4f} mm"
)


# ------------------------------------------------------------
# EVENT SIGNIFICANCE
# ------------------------------------------------------------

df["event_significant"] = (
    (df["observed_rainfall_mm"] >= 1.0)
    |
    (df["control_forecast"] >= 1.0)
)


# ------------------------------------------------------------
# BUST LABEL
# ------------------------------------------------------------

df["bust"] = (
    (df["absolute_control_error"] >= error_threshold)
    &
    df["event_significant"]
).astype(int)


# ------------------------------------------------------------
# BUST TYPE
# ------------------------------------------------------------

df["bust_type"] = "normal"

missed_event = (
    (df["bust"] == 1)
    &
    (df["control_forecast"] < 1.0)
    &
    (df["observed_rainfall_mm"] >= 1.0)
)

false_alarm = (
    (df["bust"] == 1)
    &
    (df["control_forecast"] >= 1.0)
    &
    (df["observed_rainfall_mm"] < 1.0)
)

both_significant = (
    (df["bust"] == 1)
    &
    (df["control_forecast"] >= 1.0)
    &
    (df["observed_rainfall_mm"] >= 1.0)
)


df.loc[missed_event, "bust_type"] = "missed_event"

df.loc[
    false_alarm,
    "bust_type"
] = "false_alarm"

df.loc[
    both_significant,
    "bust_type"
] = "magnitude_error"


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BUST LABEL SUMMARY")
print("=" * 70)

print(
    f"Total cases:       {len(df)}"
)

print(
    f"Bust cases:        {df['bust'].sum()}"
)

print(
    f"Normal cases:      {(df['bust'] == 0).sum()}"
)

print(
    f"Bust percentage:   "
    f"{100 * df['bust'].mean():.2f}%"
)


print("\nBust types:")

print(
    df.loc[
        df["bust"] == 1,
        "bust_type"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# BUST CASES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BUST CASES")
print("=" * 70)

display_columns = [
    "forecast_issue_time",
    "target_time",
    "forecast_time_hours",
    "control_forecast",
    "ensemble_mean",
    "ensemble_spread",
    "observed_rainfall_mm",
    "absolute_control_error",
    "bust_type",
]

print(
    df.loc[
        df["bust"] == 1,
        display_columns
    ]
    .sort_values(
        "absolute_control_error",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    "Rows:",
    len(df)
)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Missing bust labels:",
    df["bust"].isna().sum()
)

print(
    "Output:",
    OUTPUT_FILE
)