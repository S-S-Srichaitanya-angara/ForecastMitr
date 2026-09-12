import xarray as xr
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FORECAST_DIR = BASE_DIR / "data" / "gefs" / "2000013000"

CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707

MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


all_data = []


for member in MEMBERS:

    file_path = (
        FORECAST_DIR
        / member
        / f"apcp_sfc_2000013000_{member}.grib2"
    )

    print(f"\nReading {member}...")

    dataset = xr.open_dataset(
        file_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "dataType": "cf" if member == "c00" else "pf"
            }
        }
    )

    rainfall = dataset["tp"].sel(
        latitude=CHENNAI_LAT,
        longitude=CHENNAI_LON,
        method="nearest"
    )

    actual_lat = float(
        dataset.latitude.sel(
            latitude=CHENNAI_LAT,
            method="nearest"
        )
    )

    actual_lon = float(
        dataset.longitude.sel(
            longitude=CHENNAI_LON,
            method="nearest"
        )
    )

    for i in range(len(dataset.step)):

        valid_time = dataset.valid_time.values[i]

        value = float(
            rainfall.isel(step=i).values
        )

        all_data.append(
            {
                "location": "Chennai",
                "latitude": actual_lat,
                "longitude": actual_lon,
                "forecast_issue_time": str(
                    dataset.time.values
                ),
                "target_time": str(valid_time),
                "lead_time_hours": (
                    dataset.step.values[i]
                    / pd.Timedelta(hours=1)
                ),
                "ensemble_member": member,
                "forecast_rainfall_mm": value
            }
        )

    dataset.close()


df = pd.DataFrame(all_data)

output_path = (
    BASE_DIR
    / "data"
    / "ensemble_chennai.csv"
)

df.to_csv(output_path, index=False)


print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Members: {df['ensemble_member'].unique()}")
print(f"Output: {output_path}")

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))