from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
)

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = (
    BASE_DIR
    / "data"
    / "gefs"
    / "2000013000"
    / "p01"
    / "apcp_sfc_2000013000_p01.grib2"
)


three_hour_records = []


with open(FILE_PATH, "rb") as file:

    record_number = 0

    while True:

        handle = codes_grib_new_from_file(file)

        if handle is None:
            break

        record_number += 1

        start_step = codes_get(handle, "startStep")
        end_step = codes_get(handle, "endStep")
        forecast_time = codes_get(handle, "forecastTime")

        if end_step - start_step == 3:

            three_hour_records.append(
                {
                    "record": record_number,
                    "forecast_time": forecast_time,
                    "start_step": start_step,
                    "end_step": end_step,
                    "step_range": codes_get(handle, "stepRange"),
                }
            )

        codes_release(handle)


print("=" * 70)
print("GEFS 3-HOUR ACCUMULATION RECORDS")
print("=" * 70)

print(f"Total GRIB records: {record_number}")
print(f"3-hour records:     {len(three_hour_records)}")

print("\nFirst 15 three-hour records:")

for row in three_hour_records[:15]:
    print(
        f"Record {row['record']:3d} | "
        f"forecastTime={row['forecast_time']:3d} | "
        f"{row['start_step']:3d}-{row['end_step']:3d} | "
        f"{row['step_range']}"
    )

print("\nLast 10 three-hour records:")

for row in three_hour_records[-10:]:
    print(
        f"Record {row['record']:3d} | "
        f"forecastTime={row['forecast_time']:3d} | "
        f"{row['start_step']:3d}-{row['end_step']:3d} | "
        f"{row['step_range']}"
    )