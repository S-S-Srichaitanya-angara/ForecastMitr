import xarray as xr
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
GEFS_DIR = BASE_DIR / "data" / "gefs"

files = list(GEFS_DIR.glob("*.grib2"))

if not files:
    print("No GRIB2 files found.")
    raise SystemExit

file_path = files[0]

print("=" * 60)
print("FILE")
print("=" * 60)
print(file_path)


dataset = xr.open_dataset(
    file_path,
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {
            "dataType": "pf"
        }
    }
)


print("\n" + "=" * 60)
print("PRECIPITATION METADATA")
print("=" * 60)

print("Variable       :", dataset["tp"].name)
print("Long name      :", dataset["tp"].attrs.get("long_name"))
print("Units          :", dataset["tp"].attrs.get("units"))
print("Step type      :", dataset["tp"].attrs.get("GRIB_stepType"))
print("Step units     :", dataset["tp"].attrs.get("GRIB_stepUnits"))
print("Start step     :", dataset["tp"].attrs.get("GRIB_startStep"))
print("End step       :", dataset["tp"].attrs.get("GRIB_endStep"))
print("Step range     :", dataset["tp"].attrs.get("GRIB_stepRange"))
print("Forecast time  :", dataset["time"].values)
print("Number         :", dataset["number"].values)
print("Total members  :", dataset["tp"].attrs.get("GRIB_totalNumber"))


print("\n" + "=" * 60)
print("STEP INFORMATION")
print("=" * 60)

for i in range(len(dataset.step)):
    step = dataset.step.values[i]
    valid_time = dataset.valid_time.values[i]

    print(
        f"{i:02d} | "
        f"step={step} | "
        f"valid_time={valid_time}"
    )


dataset.close()