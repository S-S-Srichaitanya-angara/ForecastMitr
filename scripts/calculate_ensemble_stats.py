import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "ensemble_chennai.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "ensemble_stats_chennai.csv"
)


df = pd.read_csv(INPUT_FILE)


# ---------------------------------------------------------
# Calculate statistics across the five ensemble members
# ---------------------------------------------------------

stats = (
    df
    .groupby(
        [
            "location",
            "latitude",
            "longitude",
            "forecast_issue_time",
            "target_time",
            "forecast_time_hours",
            "start_step_hours",
            "end_step_hours",
        ]
    )["forecast_rainfall_mm"]
    .agg(
        ensemble_mean="mean",
        ensemble_median="median",
        ensemble_min="min",
        ensemble_max="max",
    )
    .reset_index()
)


# ---------------------------------------------------------
# Ensemble spread
# ---------------------------------------------------------

stats["ensemble_spread"] = (
    stats["ensemble_max"]
    - stats["ensemble_min"]
)


# ---------------------------------------------------------
# Control forecast
# ---------------------------------------------------------

control = (
    df[df["ensemble_member"] == "c00"]
    [
        [
            "target_time",
            "forecast_rainfall_mm",
        ]
    ]
    .rename(
        columns={
            "forecast_rainfall_mm": "control_forecast"
        }
    )
)


stats = stats.merge(
    control,
    on="target_time",
    how="left",
)


# ---------------------------------------------------------
# Final column order
# ---------------------------------------------------------

stats = stats[
    [
        "location",
        "latitude",
        "longitude",
        "forecast_issue_time",
        "target_time",
        "forecast_time_hours",
        "start_step_hours",
        "end_step_hours",
        "control_forecast",
        "ensemble_mean",
        "ensemble_median",
        "ensemble_min",
        "ensemble_max",
        "ensemble_spread",
    ]
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

stats.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("=" * 70)
print("ENSEMBLE STATISTICS COMPLETE")
print("=" * 70)

print(f"Rows:   {len(stats)}")
print(f"Output: {OUTPUT_FILE}")


print("\nFirst 10 rows:")

print(
    stats
    .head(10)
    .to_string(index=False)
)


print("\nStatistics:")

print(
    stats[
        [
            "control_forecast",
            "ensemble_mean",
            "ensemble_min",
            "ensemble_max",
            "ensemble_spread",
        ]
    ].describe()
)