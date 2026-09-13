from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_values,
    codes_release,
)

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

GEFS_DIR = (
    BASE_DIR
    / "data"
    / "gefs"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "ensemble_chennai.csv"
)


# ============================================================
# LOCATION
# ============================================================

CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707


# ============================================================
# ENSEMBLE MEMBERS
# ============================================================

MEMBERS = [
    "c00",
    "p01",
    "p02",
    "p03",
    "p04",
]


# ============================================================
# EXTRACT DATA
# ============================================================

all_data = []

forecast_dates = sorted(
    [
        folder
        for folder in GEFS_DIR.iterdir()
        if folder.is_dir()
    ]
)


print("=" * 70)
print("GEFS ENSEMBLE EXTRACTION")
print("=" * 70)

print(f"Initialization dates found: {len(forecast_dates)}")
print(f"Ensemble members: {MEMBERS}")
print(f"Total expected files: {len(forecast_dates) * len(MEMBERS)}")


for date_folder in forecast_dates:

    date = date_folder.name

    print("\n" + "-" * 70)
    print(f"Processing initialization: {date}")
    print("-" * 70)

    for member in MEMBERS:

        file_path = (
            date_folder
            / member
            / f"apcp_sfc_{date}_{member}.grib2"
        )

        if not file_path.exists():
            print(f"{member}: FILE NOT FOUND")
            continue

        print(f"{member}: reading...")

        record_number = 0
        three_hour_count = 0

        with open(file_path, "rb") as file:

            while True:

                handle = codes_grib_new_from_file(file)

                if handle is None:
                    break

                record_number += 1

                # ------------------------------------------------
                # Keep only 3-hour accumulation records
                # ------------------------------------------------

                start_step = codes_get(
                    handle,
                    "startStep"
                )

                end_step = codes_get(
                    handle,
                    "endStep"
                )

                if end_step - start_step != 3:

                    codes_release(handle)
                    continue

                three_hour_count += 1

                # ------------------------------------------------
                # Grid information
                # ------------------------------------------------

                data_lon = codes_get(
                    handle,
                    "longitudeOfFirstGridPointInDegrees"
                )

                nx = codes_get(
                    handle,
                    "Ni"
                )

                j_increment = codes_get(
                    handle,
                    "jDirectionIncrementInDegrees"
                )

                i_increment = codes_get(
                    handle,
                    "iDirectionIncrementInDegrees"
                )

                # ------------------------------------------------
                # Find nearest GEFS grid point
                # ------------------------------------------------

                row = round(
                    (90.0 - CHENNAI_LAT)
                    / j_increment
                )

                col = round(
                    (CHENNAI_LON - data_lon)
                    / i_increment
                )

                index = row * nx + col

                # ------------------------------------------------
                # Extract rainfall
                # ------------------------------------------------

                values = codes_get_values(handle)

                rainfall = float(
                    values[index]
                )

                # ------------------------------------------------
                # Forecast timing
                # ------------------------------------------------

                data_date = codes_get(
                    handle,
                    "dataDate"
                )

                data_time = codes_get(
                    handle,
                    "dataTime"
                )

                forecast_time = codes_get(
                    handle,
                    "forecastTime"
                )

                issue_time = pd.Timestamp(
                    str(data_date)
                    + str(data_time).zfill(4)
                )

                target_time = (
                    issue_time
                    + pd.Timedelta(
                        hours=end_step
                    )
                )

                # ------------------------------------------------
                # Store record
                # ------------------------------------------------

                all_data.append(
                    {
                        "location": "Chennai",
                        "latitude": CHENNAI_LAT,
                        "longitude": CHENNAI_LON,
                        "forecast_issue_time": issue_time,
                        "target_time": target_time,
                        "forecast_time_hours": forecast_time,
                        "start_step_hours": start_step,
                        "end_step_hours": end_step,
                        "ensemble_member": member,
                        "forecast_rainfall_mm": rainfall,
                    }
                )

                codes_release(handle)

        print(
            f"{member}: "
            f"{record_number} raw records → "
            f"{three_hour_count} three-hour records"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_data)


if df.empty:

    print("\nERROR: No forecast data extracted.")
    raise SystemExit(1)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "forecast_issue_time",
        "ensemble_member",
        "target_time",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GEFS ENSEMBLE EXTRACTION COMPLETE")
print("=" * 70)

print(f"Initialization dates: {df['forecast_issue_time'].dt.strftime('%Y%m%d%H').nunique()}")
print(f"Total rows:           {len(df)}")
print(f"Output file:          {OUTPUT_FILE}")

print("\nRows per member:")
print(
    df["ensemble_member"]
    .value_counts()
    .sort_index()
)

print("\nRows per initialization:")
print(
    df.groupby(
        df["forecast_issue_time"].dt.strftime("%Y%m%d%H")
    ).size()
)

print("\nFirst 10 rows:")
print(
    df.head(10).to_string(index=False)
)