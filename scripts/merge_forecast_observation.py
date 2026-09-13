import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_FILE = BASE_DIR / "data" / "ensemble_stats_chennai.csv"
OBSERVATION_FILE = BASE_DIR / "data" / "observations_chennai.csv"
OUTPUT_FILE = BASE_DIR / "data" / "forecast_observation_chennai.csv"


print("=" * 70)
print("MERGING GEFS FORECAST + ERA5 REFERENCE")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load forecast data
# ---------------------------------------------------------

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=[
        "forecast_issue_time",
        "target_time",
    ],
)


# ---------------------------------------------------------
# 2. Load ERA5 reference data
# ---------------------------------------------------------

observation = pd.read_csv(
    OBSERVATION_FILE,
    parse_dates=["observation_time"],
)

observation = observation.rename(
    columns={
        "observation_time": "target_time"
    }
)


print(f"Forecast cases:       {len(forecast)}")
print(f"ERA5 target times:     {len(observation)}")


# ---------------------------------------------------------
# 3. Check ERA5 uniqueness
# ---------------------------------------------------------

duplicate_observations = observation.duplicated(
    "target_time"
).sum()

print(
    f"Duplicate ERA5 target times: "
    f"{duplicate_observations}"
)

if duplicate_observations != 0:
    print("\nERROR: ERA5 contains duplicate target times.")
    raise SystemExit(1)


# ---------------------------------------------------------
# 4. Merge
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
    validate="many_to_one",
)


# ---------------------------------------------------------
# 5. Calculate forecast errors
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
# 6. Validate merge
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MERGE VALIDATION")
print("=" * 70)

print(f"Forecast cases before merge: {len(forecast)}")
print(f"Cases after merge:           {len(df)}")
print(
    f"Forecast cases without ERA5: "
    f"{len(forecast) - len(df)}"
)

print(
    f"Unique target times in result: "
    f"{df['target_time'].nunique()}"
)

print(
    f"Duplicate forecast cases: "
    f"{df.duplicated(['forecast_issue_time', 'target_time']).sum()}"
)


# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MERGE COMPLETE")
print("=" * 70)

print(f"Rows:   {len(df)}")
print(f"Output: {OUTPUT_FILE}")

print("\nFirst 15 rows:")

print(
    df[
        [
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
    ]
    .head(15)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 9. Error statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FORECAST ERROR STATISTICS")
print("=" * 70)

print(
    df[
        [
            "absolute_control_error",
            "absolute_ensemble_mean_error",
        ]
    ].describe()
)