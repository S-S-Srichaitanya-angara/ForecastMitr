import cdsapi
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "observations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

AREA = [
    13.1,   # North
    80.2,   # West
    13.0,   # South
    80.3    # East
]

TIMES = [
    "00:00",
    "01:00",
    "02:00",
    "03:00",
    "04:00",
    "05:00",
    "06:00",
    "07:00",
    "08:00",
    "09:00",
    "10:00",
    "11:00",
    "12:00",
    "13:00",
    "14:00",
    "15:00",
    "16:00",
    "17:00",
    "18:00",
    "19:00",
    "20:00",
    "21:00",
    "22:00",
    "23:00",
]

print("Downloading January 30-31...")

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "total_precipitation",
        ],
        "year": "2000",
        "month": "01",
        "day": [
            "30",
            "31",
        ],
        "time": TIMES,
        "area": AREA,
        "format": "netcdf",
    },
    str(OUTPUT_DIR / "era5_chennai_2000_01.nc"),
)

print("January download complete.")

print("\nDownloading February 1-9...")

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "total_precipitation",
        ],
        "year": "2000",
        "month": "02",
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
        ],
        "time": TIMES,
        "area": AREA,
        "format": "netcdf",
    },
    str(OUTPUT_DIR / "era5_chennai_2000_02.nc"),
)

print("February download complete.")

print("\n" + "=" * 60)
print("ERA5 DOWNLOAD COMPLETE")
print("=" * 60)
print("January:", OUTPUT_DIR / "era5_chennai_2000_01.nc")
print("February:", OUTPUT_DIR / "era5_chennai_2000_02.nc")