from pathlib import Path
from datetime import datetime, timedelta
import requests
import time
import json


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://nomads.ncep.noaa.gov/"
    "cgi-bin/filter_gefs_atmos_0p25s.pl"
)

OUTPUT_ROOT = Path("data/live_gefs_timeline")
OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# Chennai region
LEFT_LON = 79.75
RIGHT_LON = 80.75
TOP_LAT = 13.75
BOTTOM_LAT = 12.75

# Ensemble subset
MEMBERS = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]

# Start with 3-hourly forecasts.
# 0-72 h gives us a strong first demo.
FORECAST_HOURS = list(
    range(3, 73, 3)
)

# Search recent dates
MAX_DAYS_BACK = 7

# Try newest cycles first
CYCLES = [
    "18",
    "12",
    "06",
    "00",
]


# ============================================================
# URL
# ============================================================

def build_url(
    date_str,
    cycle,
    member,
    forecast_hour
):

    filename = (
        f"{member}.t{cycle}z."
        f"pgrb2s.0p25."
        f"f{forecast_hour:03d}"
    )

    directory = (
        f"/gefs.{date_str}/"
        f"{cycle}/"
        f"atmos/"
        f"pgrb2sp25"
    )

    params = {
        "file": filename,

        "lev_surface": "on",
        "var_APCP": "on",

        "subregion": "",

        "leftlon": str(LEFT_LON),
        "rightlon": str(RIGHT_LON),
        "toplat": str(TOP_LAT),
        "bottomlat": str(BOTTOM_LAT),

        "dir": directory,
    }

    response = requests.Request(
        "GET",
        BASE_URL,
        params=params
    ).prepare()

    return response.url, filename


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(url, path):

    try:

        response = requests.get(
            url,
            timeout=60
        )

        if response.status_code != 200:
            return False, (
                f"HTTP {response.status_code}"
            )

        content = response.content

        if len(content) < 100:
            return False, (
                f"File too small: "
                f"{len(content)} bytes"
            )

        # Reject obvious HTML responses.
        if content[:20].lower().startswith(
            b"<html"
        ):
            return False, "HTML response"

        with open(path, "wb") as f:
            f.write(content)

        return True, f"{len(content)} bytes"

    except Exception as e:

        return False, str(e)


# ============================================================
# TEST A CYCLE
# ============================================================

def test_cycle(
    date_str,
    cycle
):

    print()
    print("=" * 70)
    print(
        f"Testing GEFS cycle "
        f"{date_str} {cycle} UTC"
    )
    print("=" * 70)

    # First test +3h control.
    url, filename = build_url(
        date_str,
        cycle,
        "gec00",
        3
    )

    test_path = (
        OUTPUT_ROOT /
        f"_test_{filename}"
    )

    success, message = download_file(
        url,
        test_path
    )

    if test_path.exists():
        test_path.unlink()

    if not success:

        print(
            f"Cycle unavailable: {message}"
        )

        return False

    print("Cycle is usable.")

    return True


# ============================================================
# DOWNLOAD COMPLETE TIMELINE
# ============================================================

def download_cycle(
    date_str,
    cycle
):

    cycle_dir = (
        OUTPUT_ROOT /
        f"{date_str}_{cycle}"
    )

    cycle_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total = (
        len(MEMBERS)
        *
        len(FORECAST_HOURS)
    )

    completed = 0

    print()
    print(
        f"Downloading {total} "
        f"GRIB subsets"
    )

    for forecast_hour in FORECAST_HOURS:

        print()
        print(
            f"========== +{forecast_hour:03d} h "
            f"=========="
        )

        for member in MEMBERS:

            url, filename = build_url(
                date_str,
                cycle,
                member,
                forecast_hour
            )

            output_path = (
                cycle_dir /
                filename
            )

            # Don't download an existing file again.
            if output_path.exists():

                print(
                    f"{member} f{forecast_hour:03d}: "
                    f"already exists"
                )

                completed += 1
                continue

            success, message = download_file(
                url,
                output_path
            )

            if not success:

                print(
                    f"{member} f{forecast_hour:03d}: "
                    f"FAILED - {message}"
                )

                return None

            completed += 1

            print(
                f"{member} f{forecast_hour:03d}: "
                f"OK ({message})"
            )

            # Avoid hammering NOMADS.
            time.sleep(0.5)

    metadata = {
        "source": "NOAA/NCEP NOMADS",
        "dataset": "GEFS 0.25 degree",
        "date": date_str,
        "cycle": cycle,
        "members": MEMBERS,
        "forecast_hours": FORECAST_HOURS,
        "region": {
            "left_lon": LEFT_LON,
            "right_lon": RIGHT_LON,
            "top_lat": TOP_LAT,
            "bottom_lat": BOTTOM_LAT,
        },
        "completed_files": completed,
        "expected_files": total,
        "downloaded_at_utc":
            datetime.utcnow().isoformat(),
    }

    metadata_path = (
        cycle_dir /
        "metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    return {
        "directory": cycle_dir,
        "metadata": metadata_path,
        "date": date_str,
        "cycle": cycle,
    }


# ============================================================
# FIND LATEST USABLE CYCLE
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "FORECASTMITR - "
        "MULTI-LEAD GEFS DOWNLOADER"
    )
    print("=" * 70)

    today = datetime.utcnow().date()

    for days_back in range(
        MAX_DAYS_BACK + 1
    ):

        date = (
            today -
            timedelta(days=days_back)
        )

        date_str = date.strftime(
            "%Y%m%d"
        )

        print()
        print(
            f"Searching date: {date_str}"
        )

        for cycle in CYCLES:

            if not test_cycle(
                date_str,
                cycle
            ):
                continue

            result = download_cycle(
                date_str,
                cycle
            )

            if result is None:

                print(
                    "Cycle download incomplete."
                )

                continue

            print()
            print("=" * 70)
            print(
                "MULTI-LEAD GEFS DOWNLOAD SUCCESS"
            )
            print("=" * 70)

            print(
                f"Date       : {result['date']}"
            )

            print(
                f"Cycle      : {result['cycle']} UTC"
            )

            print(
                f"Forecasts  : "
                f"+3 to +72 h"
            )

            print(
                f"Members    : "
                f"{len(MEMBERS)}"
            )

            print(
                f"Directory  : "
                f"{result['directory']}"
            )

            return

    raise RuntimeError(
        "Could not find a usable GEFS cycle."
    )


if __name__ == "__main__":
    main()
