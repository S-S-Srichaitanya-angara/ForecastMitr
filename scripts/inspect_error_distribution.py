import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "forecast_observation_chennai.csv"


df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "forecast_issue_time",
        "target_time",
    ],
)


# ------------------------------------------------------------
# BASIC ERROR DISTRIBUTION
# ------------------------------------------------------------

print("=" * 70)
print("FORECAST ERROR DISTRIBUTION")
print("=" * 70)

print(f"Total cases: {len(df)}")

print("\nControl absolute error percentiles:")

for percentile in [50, 75, 80, 85, 90, 95, 97, 98, 99, 100]:
    value = df["absolute_control_error"].quantile(percentile / 100)
    print(f"{percentile:>3}th percentile : {value:.4f} mm")


# ------------------------------------------------------------
# LARGEST ERRORS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 CONTROL FORECAST ERRORS")
print("=" * 70)

top_errors = (
    df.sort_values(
        "absolute_control_error",
        ascending=False
    )
    .head(20)
)

columns = [
    "forecast_issue_time",
    "target_time",
    "forecast_time_hours",
    "control_forecast",
    "ensemble_mean",
    "ensemble_spread",
    "observed_rainfall_mm",
    "control_error",
    "absolute_control_error",
]

print(
    top_errors[columns]
    .to_string(index=False)
)


# ------------------------------------------------------------
# MISSED RAINFALL EVENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSED RAINFALL EVENTS")
print("=" * 70)

# Forecast essentially dry, but observed rainfall is meaningful.
missed = df[
    (df["control_forecast"] <= 0.1) &
    (df["observed_rainfall_mm"] >= 1.0)
].copy()

print(f"Cases: {len(missed)}")

if len(missed) > 0:
    print(
        missed[
            columns
        ]
        .sort_values(
            "observed_rainfall_mm",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# FALSE ALARM EVENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FALSE ALARM EVENTS")
print("=" * 70)

# Forecast meaningful rainfall, but observed rainfall is very small.
false_alarm = df[
    (df["control_forecast"] >= 1.0) &
    (df["observed_rainfall_mm"] <= 0.1)
].copy()

print(f"Cases: {len(false_alarm)}")

if len(false_alarm) > 0:
    print(
        false_alarm[
            columns
        ]
        .sort_values(
            "control_forecast",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# LARGE ERRORS BY LEAD TIME
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ERROR BY LEAD TIME")
print("=" * 70)

lead_stats = (
    df.groupby("forecast_time_hours")[
        "absolute_control_error"
    ]
    .agg(
        count="count",
        mean="mean",
        median="median",
        max="max",
    )
)

print(lead_stats.to_string())


# ------------------------------------------------------------
# LARGE ERRORS BY MONTH
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ERROR BY MONTH")
print("=" * 70)

df["month"] = df["target_time"].dt.month

month_stats = (
    df.groupby("month")[
        "absolute_control_error"
    ]
    .agg(
        count="count",
        mean="mean",
        median="median",
        max="max",
    )
)

print(month_stats.to_string())


# ------------------------------------------------------------
# VERY LARGE ERRORS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LARGE ERROR COUNTS")
print("=" * 70)

for threshold in [0.5, 1.0, 2.0, 5.0, 10.0]:
    count = (
        df["absolute_control_error"] >= threshold
    ).sum()

    percentage = 100 * count / len(df)

    print(
        f">= {threshold:>4.1f} mm : "
        f"{count:>3} cases "
        f"({percentage:.2f}%)"
    )