import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "ensemble_chennai.csv"
OUTPUT_FILE = BASE_DIR / "data" / "ensemble_stats_chennai.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


# ---------------------------------------------------------
# Separate control and perturbed members
# ---------------------------------------------------------

control = df[df["ensemble_member"] == "c00"].copy()

perturbed = df[df["ensemble_member"].isin(
    ["p01", "p02", "p03", "p04"]
)].copy()


# ---------------------------------------------------------
# Calculate ensemble statistics
# ---------------------------------------------------------

stats = (
    perturbed
    .groupby(
        [
            "location",
            "latitude",
            "longitude",
            "forecast_issue_time",
            "target_time",
            "lead_time_hours"
        ]
    )["forecast_rainfall_mm"]
    .agg(
        ensemble_mean="mean",
        ensemble_median="median",
        ensemble_min="min",
        ensemble_max="max"
    )
    .reset_index()
)


# Spread = maximum - minimum

stats["ensemble_spread"] = (
    stats["ensemble_max"]
    - stats["ensemble_min"]
)


# ---------------------------------------------------------
# Add control forecast
# ---------------------------------------------------------

control = control[
    [
        "target_time",
        "forecast_rainfall_mm"
    ]
].rename(
    columns={
        "forecast_rainfall_mm": "control_forecast"
    }
)


stats = stats.merge(
    control,
    on="target_time",
    how="left"
)


# ---------------------------------------------------------
# Arrange columns
# ---------------------------------------------------------

stats = stats[
    [
        "location",
        "latitude",
        "longitude",
        "forecast_issue_time",
        "target_time",
        "lead_time_hours",
        "control_forecast",
        "ensemble_mean",
        "ensemble_median",
        "ensemble_min",
        "ensemble_max",
        "ensemble_spread"
    ]
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

stats.to_csv(
    OUTPUT_FILE,
    index=False
)


print("=" * 60)
print("ENSEMBLE STATISTICS COMPLETE")
print("=" * 60)

print(f"Rows: {len(stats)}")
print(f"Output: {OUTPUT_FILE}")

print("\nFirst 10 rows:")
print(stats.head(10).to_string(index=False))