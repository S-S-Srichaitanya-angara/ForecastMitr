"""
ForecastMitr — One-click 2018 dataset downloader + cleanup

Run from the ForecastMitr project root:

    python scripts/download_2018_dataset.py

What it does:
1. Discovers all available 2018 GEFSv12 reforecast initialization folders.
2. Downloads only c00 + p01-p04 precipitation files.
3. Keeps the downloaded files in data/gefs/2018/.
4. Removes incomplete/empty downloads when detected.
5. Reuses files that are already complete.
6. Extracts Chennai 3-hour precipitation into data/ensemble_chennai_2018.csv.
7. Downloads the required 2018 ERA5 monthly precipitation files.
8. Builds data/observations_chennai_2018.csv.
9. Merges forecast + ERA5 reference into data/forecast_observation_chennai_2018.csv.
10. Creates a clean ML-ready CSV.

Notes:
- GEFS precipitation is the 3-hour accumulation message only.
- Location is the nearest grid point to Chennai (13.0827, 80.2707).
- ERA5 is used as a historical reference, not direct rain-gauge ground truth.
- This script does NOT download every GEFS variable/member; only c00+p01-p04 APCP.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GEFS_DIR = DATA_DIR / "gefs" / "2018"
OBS_DIR = DATA_DIR / "observations"

for d in [GEFS_DIR, OBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Keep these small enough for the MVP.
MEMBERS = ["c00", "p01", "p02", "p03", "p04"]

LAT = 13.0827
LON = 80.2707

BUCKET = "https://noaa-gefs-retrospective.s3.amazonaws.com"

try:
    from eccodes import (
        codes_grib_new_from_file,
        codes_get,
        codes_get_values,
        codes_release,
    )
except ImportError:
    print("ERROR: eccodes is not installed.")
    print("Run: pip install eccodes")
    sys.exit(1)


def run(cmd):
    print(">", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def download(url, destination):
    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(destination)


def discover_dates():
    """
    S3 bucket listing is paginated. We use the public S3 XML listing
    and collect 2018/initialization prefixes.
    """
    import xml.etree.ElementTree as ET

    dates = set()
    token = None

    while True:
        if token:
            url = (
                f"{BUCKET}/?list-type=2&delimiter=/"
                f"&prefix=GEFSv12/reforecast/2018/"
                f"&continuation-token={urllib.parse.quote(token)}"
            )
        else:
            url = (
                f"{BUCKET}/?list-type=2&delimiter=/"
                f"&prefix=GEFSv12/reforecast/2018/"
            )

        print("Listing:", url)
        xml_data = urllib.request.urlopen(url).read()
        root = ET.fromstring(xml_data)

        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        for prefix in root.findall("s3:CommonPrefixes/s3:Prefix", ns):
            value = prefix.text or ""
            parts = value.strip("/").split("/")
            if len(parts) >= 4:
                date = parts[3]
                if len(date) == 10 and date.isdigit():
                    dates.add(date)

        truncated = root.find("s3:IsTruncated", ns)
        if truncated is None or truncated.text != "true":
            break

        next_token = root.find("s3:NextContinuationToken", ns)
        if next_token is None:
            break

        token = next_token.text

    return sorted(dates)


def get_grid_index(handle):
    first_lon = codes_get(
        handle, "longitudeOfFirstGridPointInDegrees"
    )
    nx = codes_get(handle, "Ni")
    j_inc = codes_get(
        handle, "jDirectionIncrementInDegrees"
    )
    i_inc = codes_get(
        handle, "iDirectionIncrementInDegrees"
    )

    row = round((90.0 - LAT) / j_inc)
    col = round((LON - first_lon) / i_inc)

    return row * nx + col


def extract_file(file_path, member):
    rows = []

    with open(file_path, "rb") as f:
        while True:
            handle = codes_grib_new_from_file(f)

            if handle is None:
                break

            try:
                start = codes_get(handle, "startStep")
                end = codes_get(handle, "endStep")

                # Keep only 3-hour accumulation records.
                if end - start != 3:
                    continue

                index = get_grid_index(handle)
                values = codes_get_values(handle)

                rainfall = float(values[index])

                data_date = codes_get(handle, "dataDate")
                data_time = codes_get(handle, "dataTime")

                import pandas as pd

                issue_time = pd.Timestamp(
                    str(data_date)
                    + str(data_time).zfill(4)
                )

                target_time = (
                    issue_time
                    + pd.Timedelta(hours=end)
                )

                rows.append({
                    "location": "Chennai",
                    "latitude": LAT,
                    "longitude": LON,
                    "forecast_issue_time": issue_time,
                    "target_time": target_time,
                    "forecast_time_hours": codes_get(
                        handle, "forecastTime"
                    ),
                    "start_step_hours": start,
                    "end_step_hours": end,
                    "ensemble_member": member,
                    "forecast_rainfall_mm": rainfall,
                })

            finally:
                codes_release(handle)

    return rows


def main():
    import urllib.parse
    import pandas as pd

    print("=" * 75)
    print("FORECASTMITR — ONE-CLICK 2018 DATASET BUILDER")
    print("=" * 75)

    print("\nDiscovering 2018 GEFSv12 initialization dates...")
    dates = discover_dates()

    if not dates:
        raise RuntimeError(
            "No 2018 GEFS initialization dates were found."
        )

    print(f"Found {len(dates)} initialization dates.")
    print(f"Members: {MEMBERS}")
    print(f"Expected GRIB files: {len(dates) * len(MEMBERS)}")

    # ---------------------------------------------------------
    # DOWNLOAD GEFS
    # ---------------------------------------------------------

    downloaded = 0
    reused = 0
    failed = []

    for i, date in enumerate(dates, start=1):
        print("\n" + "-" * 75)
        print(f"[{i}/{len(dates)}] Initialization {date}")
        print("-" * 75)

        for member in MEMBERS:
            folder = GEFS_DIR / date / member
            folder.mkdir(parents=True, exist_ok=True)

            filename = f"apcp_sfc_{date}_{member}.grib2"
            destination = folder / filename

            url = (
                f"{BUCKET}/GEFSv12/reforecast/"
                f"2018/{date}/{member}/Days:1-10/{filename}"
            )

            if destination.exists() and destination.stat().st_size > 10000:
                print(f"{member}: already exists")
                reused += 1
                continue

            try:
                download(url, destination)
                downloaded += 1
                print(
                    f"{member}: downloaded "
                    f"({destination.stat().st_size / 1024 / 1024:.1f} MB)"
                )
            except Exception as exc:
                if destination.exists():
                    destination.unlink()
                failed.append((date, member, str(exc)))
                print(f"{member}: FAILED — {exc}")

    print("\nGEFS download summary:")
    print(f"Downloaded: {downloaded}")
    print(f"Reused:     {reused}")
    print(f"Failed:     {len(failed)}")

    if failed:
        print("\nFailed files:")
        for item in failed[:20]:
            print(item)

        if len(failed) == len(dates) * len(MEMBERS):
            raise RuntimeError("All GEFS downloads failed.")

    # ---------------------------------------------------------
    # EXTRACT ENSEMBLE
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("EXTRACTING CHENNAI GEFS DATA")
    print("=" * 75)

    all_rows = []

    for date in dates:
        for member in MEMBERS:
            file_path = (
                GEFS_DIR
                / date
                / member
                / f"apcp_sfc_{date}_{member}.grib2"
            )

            if not file_path.exists():
                print(f"SKIP missing: {file_path}")
                continue

            try:
                rows = extract_file(file_path, member)
                all_rows.extend(rows)
                print(
                    f"{date} {member}: "
                    f"{len(rows)} three-hour records"
                )
            except Exception as exc:
                print(
                    f"ERROR reading {file_path}: {exc}"
                )

    ensemble = pd.DataFrame(all_rows)

    if ensemble.empty:
        raise RuntimeError(
            "No GEFS records were extracted."
        )

    ensemble = ensemble.sort_values(
        [
            "forecast_issue_time",
            "target_time",
            "ensemble_member",
        ]
    ).reset_index(drop=True)

    ensemble_file = DATA_DIR / "ensemble_chennai_2018.csv"
    ensemble.to_csv(ensemble_file, index=False)

    print(
        f"\nSaved {len(ensemble)} raw ensemble rows:"
        f"\n{ensemble_file}"
    )

    # ---------------------------------------------------------
    # ENSEMBLE STATISTICS
    # ---------------------------------------------------------

    grouped = ensemble.groupby(
        [
            "forecast_issue_time",
            "target_time",
        ]
    )

    stats = grouped["forecast_rainfall_mm"].agg(
        ensemble_mean="mean",
        ensemble_median="median",
        ensemble_min="min",
        ensemble_max="max",
    ).reset_index()

    control = ensemble[
        ensemble["ensemble_member"] == "c00"
    ][
        [
            "forecast_issue_time",
            "target_time",
            "forecast_rainfall_mm",
        ]
    ].rename(
        columns={
            "forecast_rainfall_mm":
            "control_forecast"
        }
    )

    stats = stats.merge(
        control,
        on=[
            "forecast_issue_time",
            "target_time",
        ],
        how="inner",
        validate="one_to_one",
    )

    stats["ensemble_spread"] = (
        stats["ensemble_max"]
        - stats["ensemble_min"]
    )

    stats["forecast_time_hours"] = (
        (
            pd.to_datetime(stats["target_time"])
            - pd.to_datetime(stats["forecast_issue_time"])
        ).dt.total_seconds()
        / 3600
    )

    stats_file = DATA_DIR / "ensemble_stats_chennai_2018.csv"
    stats.to_csv(stats_file, index=False)

    print(f"Saved ensemble statistics: {stats_file}")
    print(f"Forecast cases: {len(stats)}")

    # ---------------------------------------------------------
    # ERA5 MONTHLY DOWNLOAD
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("ERA5 REFERENCE DATA")
    print("=" * 75)

    print(
        "ERA5 download requires a working CDS API configuration."
    )

    try:
        import cdsapi
    except ImportError:
        raise RuntimeError(
            "cdsapi is not installed. Run: pip install cdsapi"
        )

    client = cdsapi.Client()

    target_times = pd.to_datetime(
        stats["target_time"]
    )

    months = sorted(
        target_times.dt.month.unique()
    )

    for month in months:
        output = (
            OBS_DIR
            / f"era5_chennai_2018_{month:02d}.nc"
        )

        if output.exists() and output.stat().st_size > 100000:
            print(f"ERA5 {month:02d}: already exists")
            continue

        print(f"Downloading ERA5 month {month:02d}...")

        days = [
            f"{day:02d}"
            for day in range(1, 32)
        ]

        # Remove invalid dates for short months.
        last_day = (
            pd.Timestamp(2018, month, 1)
            + pd.offsets.MonthEnd(0)
        ).day

        days = [
            f"{day:02d}"
            for day in range(1, last_day + 1)
        ]

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "total_precipitation"
                ],
                "year": "2018",
                "month": f"{month:02d}",
                "day": days,
                "time": [
                    f"{hour:02d}:00"
                    for hour in range(24)
                ],
                "area": [
                    13.1,
                    80.2,
                    13.0,
                    80.3,
                ],
                "format": "netcdf",
            },
            str(output),
        )

        print(f"ERA5 {month:02d}: complete")

    # ---------------------------------------------------------
    # BUILD OBSERVATIONS
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("BUILDING ERA5 TARGET REFERENCES")
    print("=" * 75)

    import xarray as xr

    nc_files = sorted(
        OBS_DIR.glob("era5_chennai_2018_*.nc")
    )

    if not nc_files:
        raise RuntimeError("No ERA5 files found.")

    datasets = [
        xr.open_dataset(file)
        for file in nc_files
    ]

    era5 = xr.concat(
        datasets,
        dim="valid_time",
    )

    # Nearest ERA5 point.
    point = era5.sel(
        latitude=LAT,
        longitude=LON,
        method="nearest",
    )

    series = point["tp"].to_series()

    series.index = pd.to_datetime(series.index)

    observations = []

    for target in target_times:
        hours = [
            target - pd.Timedelta(hours=2),
            target - pd.Timedelta(hours=1),
            target,
        ]

        values = series.reindex(hours)

        if values.isna().any():
            observed = float("nan")
        else:
            # ERA5 total precipitation is metres.
            # Convert to mm.
            observed = float(values.sum() * 1000)

        observations.append({
            "target_time": target,
            "observed_rainfall_mm": observed,
        })

    obs = pd.DataFrame(observations)

    if obs["observed_rainfall_mm"].isna().any():
        missing = obs["observed_rainfall_mm"].isna().sum()
        raise RuntimeError(
            f"{missing} target times have missing ERA5 data."
        )

    obs_file = DATA_DIR / "observations_chennai_2018.csv"
    obs.to_csv(obs_file, index=False)

    print(f"Saved observations: {obs_file}")

    # ---------------------------------------------------------
    # FINAL MERGE
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("BUILDING FINAL 2018 DATASET")
    print("=" * 75)

    final = stats.merge(
        obs,
        on="target_time",
        how="inner",
        validate="many_to_one",
    )

    final["control_error"] = (
        final["control_forecast"]
        - final["observed_rainfall_mm"]
    )

    final["absolute_control_error"] = (
        final["control_error"].abs()
    )

    final["ensemble_mean_error"] = (
        final["ensemble_mean"]
        - final["observed_rainfall_mm"]
    )

    final["absolute_ensemble_mean_error"] = (
        final["ensemble_mean_error"].abs()
    )

    final["month"] = pd.to_datetime(
        final["target_time"]
    ).dt.month

    final["control_vs_ensemble"] = (
        final["control_forecast"]
        - final["ensemble_mean"]
    )

    final["relative_spread"] = (
        final["ensemble_spread"]
        / (final["ensemble_mean"] + 0.1)
    )

    final_file = (
        DATA_DIR
        / "forecast_observation_chennai_2018.csv"
    )

    final.to_csv(final_file, index=False)

    print(f"Final rows: {len(final)}")
    print(f"Final dataset: {final_file}")

    print("\n" + "=" * 75)
    print("2018 DATASET BUILD COMPLETE")
    print("=" * 75)

    print(f"Initialization dates: {len(dates)}")
    print(f"Forecast cases:       {len(stats)}")
    print(f"Final merged cases:    {len(final)}")
    print(
        f"GEFS files expected:  {len(dates) * len(MEMBERS)}"
    )
    print(
        f"GEFS files downloaded/reused: "
        f"{downloaded + reused}"
    )

    print("\nNext step:")
    print(
        "Create leakage-free bust labels and perform "
        "chronological train/test evaluation."
    )


if __name__ == "__main__":
    main()