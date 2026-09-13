import pandas as pd
import xarray as xr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_FILE = BASE_DIR / "data" / "ensemble_stats_chennai.csv"
OBSERVATION_DIR = BASE_DIR / "data" / "observations"
OUTPUT_FILE = BASE_DIR / "data" / "observations_chennai.csv"

CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707


print("=" * 70)
print("BUILDING ERA5 REFERENCE RAINFALL")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load GEFS target times
# ---------------------------------------------------------

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=["target_time"]
)

target_times = (
    forecast["target_time"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

print(f"GEFS target times: {len(target_times)}")
print(f"Earliest target:   {target_times.min()}")
print(f"Latest target:      {target_times.max()}")


# ---------------------------------------------------------
# 2. Find all ERA5 monthly files
# ---------------------------------------------------------

era5_files = sorted(
    OBSERVATION_DIR.glob("era5_chennai_2000_*.nc")
)

if not era5_files:
    print("\nERROR: No ERA5 files found.")
    raise SystemExit(1)

print(f"\nERA5 files found: {len(era5_files)}")

for file in era5_files:
    print(f"  {file.name}")


# ---------------------------------------------------------
# 3. Load ERA5 files
# ---------------------------------------------------------

datasets = []

for file in era5_files:
    print(f"\nReading: {file.name}")

    dataset = xr.open_dataset(file)

    print("Variables:", list(dataset.data_vars))

    datasets.append(dataset)


# ---------------------------------------------------------
# 4. Combine all months
# ---------------------------------------------------------

print("\nCombining ERA5 monthly datasets...")

dataset = xr.concat(
    datasets,
    dim="valid_time"
)

print("Combined dataset:")
print(dataset)


# ---------------------------------------------------------
# 5. Select Chennai grid point
# ---------------------------------------------------------

tp = dataset["tp"]

print("\nSelecting nearest ERA5 grid point to Chennai...")

tp_chennai = tp.sel(
    latitude=CHENNAI_LAT,
    longitude=CHENNAI_LON,
    method="nearest"
)

print(
    "Selected location:",
    float(tp_chennai.latitude.values),
    float(tp_chennai.longitude.values)
)


# ---------------------------------------------------------
# 6. Convert ERA5 precipitation
# ---------------------------------------------------------

# ERA5 total precipitation is stored in metres.
# Convert metres → millimetres.

hourly = (
    tp_chennai
    .to_series()
    .rename("hourly_rainfall_mm")
    .reset_index()
)

hourly["hourly_rainfall_mm"] *= 1000.0

hourly["valid_time"] = pd.to_datetime(
    hourly["valid_time"]
)

hourly = hourly[
    ["valid_time", "hourly_rainfall_mm"]
]

hourly = hourly.sort_values(
    "valid_time"
).reset_index(drop=True)

print("\nHourly ERA5 records:", len(hourly))

print("\nERA5 rainfall statistics:")
print(
    hourly["hourly_rainfall_mm"].describe()
)


# ---------------------------------------------------------
# 7. Create 3-hour reference rainfall
# ---------------------------------------------------------

print("\nCreating 3-hour rainfall totals...")

results = []

hourly_lookup = hourly.set_index("valid_time")[
    "hourly_rainfall_mm"
]

missing_targets = []

for target_time in target_times:

    required_times = pd.date_range(
        end=target_time,
        periods=3,
        freq="h"
    )

    missing = [
        timestamp
        for timestamp in required_times
        if timestamp not in hourly_lookup.index
    ]

    if missing:
        missing_targets.append(
            (target_time, missing)
        )
        continue

    rainfall = hourly_lookup.loc[
        required_times
    ].sum()

    results.append({
        "observation_time": target_time,
        "observed_rainfall_mm": float(rainfall),
    })


# ---------------------------------------------------------
# 8. Create output dataframe
# ---------------------------------------------------------

observations = pd.DataFrame(results)

observations = observations.sort_values(
    "observation_time"
).reset_index(drop=True)


# ---------------------------------------------------------
# 9. Save
# ---------------------------------------------------------

observations.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 10. Validation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ERA5 REFERENCE DATA COMPLETE")
print("=" * 70)

print(f"Expected target times: {len(target_times)}")
print(f"Reference records:     {len(observations)}")
print(f"Missing targets:       {len(missing_targets)}")

if missing_targets:
    print("\nWARNING: Missing target times:")

    for target_time, missing in missing_targets[:20]:
        print(
            f"{target_time}: missing {missing}"
        )

else:
    print("\nSUCCESS: Every GEFS target time has ERA5 reference data.")


print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 15 observations:")

print(
    observations
    .head(15)
    .to_string(index=False)
)

print("\nReference rainfall statistics:")

print(
    observations[
        "observed_rainfall_mm"
    ].describe()
)


# ---------------------------------------------------------
# 11. Close datasets
# ---------------------------------------------------------

for dataset_item in datasets:
    dataset_item.close()