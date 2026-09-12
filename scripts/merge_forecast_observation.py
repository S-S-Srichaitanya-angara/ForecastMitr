import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_FILE = (
    BASE_DIR
    / "data"
    / "ensemble_stats_chennai.csv"
)

OBSERVATION_FILE = (
    BASE_DIR
    / "data"
    / "observations_chennai.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "forecast_observation_chennai.csv"
)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=[
        "forecast_issue_time",
        "target_time",
    ],
)

observation = pd.read_csv(
    OBSERVATION_FILE,
    parse_dates=[
        "observation_time",
    ],
)


# ---------------------------------------------------------
# Rename observation time to target_time
# ---------------------------------------------------------

observation = observation.rename(
    columns={
        "observation_time": "target_time",
    }
)


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

df = forecast.merge(
    observation[
        [
            "target_time",
            "observed_rainfall_mm",
        ]
    ],
    on="target_time",
    how="inner",
)


# ---------------------------------------------------------
# Calculate errors
# ---------------------------------------------------------

df["control_error"] = (
    df["control_forecast"]
    - df["observed_rainfall_mm"]
)

df["ensemble_mean_error"] = (
    df["ensemble_mean"]
    - df["observed_rainfall_mm"]
)

df["absolute_control_error"] = (
    df["control_error"].abs()
)

df["absolute_ensemble_mean_error"] = (
    df["ensemble_mean_error"].abs()
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("=" * 70)
print("FORECAST + OBSERVATION MERGE COMPLETE")
print("=" * 70)

print(f"Rows:   {len(df)}")
print(f"Output: {OUTPUT_FILE}")


print("\nFirst 15 rows:")

print(
    df[
        [
            "target_time",
            "control_forecast",
            "ensemble_mean",
            "ensemble_spread",
            "observed_rainfall_mm",
            "control_error",
            "ensemble_mean_error",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print("\nError statistics:")

print(
    df[
        [
            "absolute_control_error",
            "absolute_ensemble_mean_error",
        ]
    ].describe()
)