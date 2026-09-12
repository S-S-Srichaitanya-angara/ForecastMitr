import pandas as pd
import xarray as xr
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

ERA5_DIR = (
    BASE_DIR
    / "data"
    / "observations"
)

FORECAST_FILE = (
    BASE_DIR
    / "data"
    / "ensemble_stats_chennai.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "observations_chennai.csv"
)


# ---------------------------------------------------------
# Load GEFS target times
# ---------------------------------------------------------

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=["target_time"]
)

target_times = (
    forecast["target_time"]
    .drop_duplicates()
    .sort_values()
)


# ---------------------------------------------------------
# Load ERA5 files
# ---------------------------------------------------------

files = [
    ERA5_DIR / "era5_chennai_2000_01.nc",
    ERA5_DIR / "era5_chennai_2000_02.nc",
]


datasets = []

for file_path in files:

    print(f"Reading {file_path.name}...")

    dataset = xr.open_dataset(file_path)

    datasets.append(dataset)


era5 = xr.concat(
    datasets,
    dim="valid_time"
)


# ---------------------------------------------------------
# Extract hourly precipitation
# ---------------------------------------------------------

tp = era5["tp"]


# Convert metres → millimetres

hourly = (
    tp
    .to_series()
    .rename("hourly_rainfall_mm")
    .reset_index()
)


hourly["hourly_rainfall_mm"] *= 1000


# Keep only time and rainfall

hourly = hourly[
    [
        "valid_time",
        "hourly_rainfall_mm",
    ]
]


hourly["valid_time"] = pd.to_datetime(
    hourly["valid_time"]
)


hourly = hourly.sort_values(
    "valid_time"
)


# ---------------------------------------------------------
# Construct the GEFS-matching 3-hour accumulation
# ---------------------------------------------------------

observations = []


for target_time in target_times:

    target_time = pd.Timestamp(target_time)

    hour_1 = target_time - pd.Timedelta(hours=2)
    hour_2 = target_time - pd.Timedelta(hours=1)
    hour_3 = target_time

    required_times = [
        hour_1,
        hour_2,
        hour_3,
    ]

    subset = hourly[
        hourly["valid_time"].isin(required_times)
    ]

    if len(subset) != 3:

        print(
            f"WARNING: Missing ERA5 data for "
            f"{target_time}"
        )

        continue

    rainfall = subset[
        "hourly_rainfall_mm"
    ].sum()

    observations.append(
        {
            "location": "Chennai",
            "latitude": 13.0827,
            "longitude": 80.2707,
            "observation_time": target_time,
            "observed_rainfall_mm": rainfall,
        }
    )


observations_df = pd.DataFrame(
    observations
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

observations_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)
print("ERA5 3-HOUR REFERENCE COMPLETE")
print("=" * 70)

print(
    f"Rows:   {len(observations_df)}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print("\nFirst 10 rows:")

print(
    observations_df
    .head(10)
    .to_string(index=False)
)

print("\nRainfall statistics:")

print(
    observations_df[
        "observed_rainfall_mm"
    ].describe()
)


era5.close()

for dataset in datasets:
    dataset.close()