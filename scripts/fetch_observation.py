import cdsapi
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_FILE = (
    BASE_DIR
    / "data"
    / "ensemble_stats_chennai.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "observations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHENNAI AREA
# ============================================================

AREA = [
    13.1,   # North
    80.2,   # West
    13.0,   # South
    80.3    # East
]


# ============================================================
# READ FORECAST TARGET TIMES
# ============================================================

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=["target_time"]
)

target_times = forecast["target_time"]


# ============================================================
# DETERMINE REQUIRED MONTHS
# ============================================================

required_months = sorted(
    target_times.dt.to_period("M").unique()
)

print("=" * 70)
print("ERA5 REFERENCE DATA DOWNLOAD")
print("=" * 70)

print(f"Required months: {len(required_months)}")

for month in required_months:
    print(f"  {month}")


# ============================================================
# ERA5 CLIENT
# ============================================================

client = cdsapi.Client()


# ============================================================
# DOWNLOAD MONTH BY MONTH
# ============================================================

for month in required_months:

    year = str(month.year)
    month_number = f"{month.month:02d}"

    output_file = (
        OUTPUT_DIR
        / f"era5_chennai_{year}_{month_number}.nc"
    )

    print("\n" + "-" * 70)
    print(f"Processing {year}-{month_number}")
    print("-" * 70)

    # --------------------------------------------------------
    # Skip existing files
    # --------------------------------------------------------

    if output_file.exists():

        print(
            f"File already exists → skipping:\n"
            f"{output_file}"
        )

        continue

    # --------------------------------------------------------
    # All days in this month
    # --------------------------------------------------------

    start_date = pd.Timestamp(
        year + "-" + month_number + "-01"
    )

    end_date = (
        start_date
        + pd.offsets.MonthEnd(1)
    )

    days = [
        f"{day:02d}"
        for day in range(
            1,
            end_date.day + 1
        )
    ]

    # --------------------------------------------------------
    # All hours
    # --------------------------------------------------------

    times = [
        f"{hour:02d}:00"
        for hour in range(24)
    ]

    print(
        f"Downloading {year}-{month_number} "
        f"({len(days)} days × 24 hours)"
    )

    # --------------------------------------------------------
    # CDS request
    # --------------------------------------------------------

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "total_precipitation"
            ],
            "year": year,
            "month": month_number,
            "day": days,
            "time": times,
            "area": AREA,
            "format": "netcdf",
        },
        str(output_file),
    )

    print("Download complete.")
    print(f"Saved to: {output_file}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERA5 DOWNLOAD COMPLETE")
print("=" * 70)

files = sorted(
    OUTPUT_DIR.glob(
        "era5_chennai_2000_*.nc"
    )
)

print(f"ERA5 files available: {len(files)}")

for file in files:
    print(f"  {file.name}")