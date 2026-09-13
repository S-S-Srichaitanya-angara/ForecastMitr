import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "ensemble_chennai.csv"
OUTPUT_FILE = BASE_DIR / "data" / "ensemble_stats_chennai.csv"


# ---------------------------------------------------------
# 1. Load raw ensemble data
# ---------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "forecast_issue_time",
        "target_time",
    ],
)


print("=" * 70)
print("CALCULATING ENSEMBLE STATISTICS")
print("=" * 70)

print(f"Raw rows: {len(df)}")


# ---------------------------------------------------------
# 2. Validate raw keys
# ---------------------------------------------------------

duplicate_keys = df.duplicated(
    [
        "forecast_issue_time",
        "target_time",
        "ensemble_member",
    ]
).sum()

print(
    f"Duplicate issue/target/member keys: "
    f"{duplicate_keys}"
)

if duplicate_keys != 0:
    print("\nERROR: Duplicate forecast records found.")
    raise SystemExit(1)


# ---------------------------------------------------------
# 3. Calculate ensemble statistics
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
# 4. Calculate ensemble spread
# ---------------------------------------------------------

stats["ensemble_spread"] = (
    stats["ensemble_max"]
    - stats["ensemble_min"]
)


# ---------------------------------------------------------
# 5. Extract control forecast
# ---------------------------------------------------------

control = (
    df[df["ensemble_member"] == "c00"]
    [
        [
            "forecast_issue_time",
            "target_time",
            "forecast_rainfall_mm",
        ]
    ]
    .rename(
        columns={
            "forecast_rainfall_mm":
                "control_forecast"
        }
    )
)


# ---------------------------------------------------------
# 6. Validate control records
# ---------------------------------------------------------

duplicate_control_keys = control.duplicated(
    [
        "forecast_issue_time",
        "target_time",
    ]
).sum()

print(
    f"Duplicate control keys: "
    f"{duplicate_control_keys}"
)

if duplicate_control_keys != 0:
    print("\nERROR: Duplicate control forecasts found.")
    raise SystemExit(1)


# ---------------------------------------------------------
# 7. Merge control correctly
# ---------------------------------------------------------

stats = stats.merge(
    control,
    on=[
        "forecast_issue_time",
        "target_time",
    ],
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# 8. Validate final statistics
# ---------------------------------------------------------

if len(stats) != 840:
    print(
        f"\nWARNING: Expected 840 cases, "
        f"but obtained {len(stats)}."
    )

if stats["control_forecast"].isna().any():
    print("\nERROR: Missing control forecasts.")
    raise SystemExit(1)


# ---------------------------------------------------------
# 9. Arrange columns
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
# 10. Save
# ---------------------------------------------------------

stats.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# 11. Final validation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ENSEMBLE STATISTICS COMPLETE")
print("=" * 70)

print(f"Rows: {len(stats)}")

print(
    "Unique forecast/target pairs:",
    stats[
        [
            "forecast_issue_time",
            "target_time",
        ]
    ].drop_duplicates().shape[0],
)

print(
    "Duplicate forecast/target pairs:",
    stats.duplicated(
        [
            "forecast_issue_time",
            "target_time",
        ]
    ).sum(),
)

print("\nRows per initialization:")

print(
    stats
    .groupby("forecast_issue_time")
    .size()
    .to_string()
)

print("\nLead-time distribution:")

print(
    stats[
        "forecast_time_hours"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nFirst 10 rows:")

print(
    stats
    .head(10)
    .to_string(index=False)
)

print("\nOutput:")
print(OUTPUT_FILE)