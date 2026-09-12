from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_values,
    codes_release,
)

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_DIR = (
    BASE_DIR
    / "data"
    / "gefs"
    / "2000013000"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "ensemble_chennai.csv"
)


CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707

MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


all_data = []


for member in MEMBERS:

    print(f"\nReading {member}...")

    file_path = (
        FORECAST_DIR
        / member
        / f"apcp_sfc_2000013000_{member}.grib2"
    )

    with open(file_path, "rb") as file:

        record_number = 0
        three_hour_count = 0

        while True:

            handle = codes_grib_new_from_file(file)

            if handle is None:
                break

            record_number += 1

            start_step = codes_get(handle, "startStep")
            end_step = codes_get(handle, "endStep")

            # Keep only 3-hour accumulation records
            if end_step - start_step != 3:
                codes_release(handle)
                continue

            three_hour_count += 1

            data_lat = codes_get(handle, "latitudeOfFirstGridPointInDegrees")
            data_lon = codes_get(handle, "longitudeOfFirstGridPointInDegrees")

            nx = codes_get(handle, "Ni")
            ny = codes_get(handle, "Nj")

            i_increment = codes_get(
                handle,
                "iDirectionIncrementInDegrees"
            )

            j_increment = codes_get(
                handle,
                "jDirectionIncrementInDegrees"
            )

            values = codes_get_values(handle)

            # GEFS grid starts at 90 N and goes southward.
            # Longitude starts at 0 E.
            row = round(
                (90.0 - CHENNAI_LAT) / j_increment
            )

            col = round(
                (CHENNAI_LON - data_lon) / i_increment
            )

            index = row * nx + col

            rainfall = float(values[index])

            data_date = codes_get(handle, "dataDate")
            data_time = codes_get(handle, "dataTime")

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
                + pd.Timedelta(hours=end_step)
            )

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

        print(f"Total records:      {record_number}")
        print(f"3-hour records:     {three_hour_count}")


df = pd.DataFrame(all_data)


df = df.sort_values(
    [
        "ensemble_member",
        "target_time",
    ]
).reset_index(drop=True)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("ENSEMBLE EXTRACTION COMPLETE")
print("=" * 70)

print(f"Rows:   {len(df)}")
print(f"Output: {OUTPUT_FILE}")

print("\nRows per member:")
print(
    df["ensemble_member"]
    .value_counts()
    .sort_index()
)

print("\nFirst 15 rows:")
print(
    df.head(15).to_string(index=False)
)