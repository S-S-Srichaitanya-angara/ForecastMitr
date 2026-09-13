from pathlib import Path
import requests
import time


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://nomads.ncep.noaa.gov/"
    "cgi-bin/filter_gefs_atmos_0p25s.pl"
)

DATE = "20260913"
CYCLE = "00"
FORECAST_HOUR = 3

MEMBERS = [
    "c00",
    "p01",
    "p02",
    "p03",
    "p04",
]

# Chennai region
LEFT_LON = 79.75
RIGHT_LON = 80.75
TOP_LAT = 13.75
BOTTOM_LAT = 12.75

OUTPUT_DIR = Path("data/live_gefs_test")


# ============================================================
# DOWNLOAD ONE MEMBER
# ============================================================

def download_member(member):

    filename = (
        f"{'gec00' if member == 'c00' else 'gep' + member[1:]}"
        f".t{CYCLE}z.pgrb2s.0p25.f{FORECAST_HOUR:03d}"
    )

    params = {
        "file": filename,

        # Only precipitation
        "var_APCP": "on",

        # Surface level
        "lev_surface": "on",

        # Geographic subset
        "subregion": "",
        "leftlon": LEFT_LON,
        "rightlon": RIGHT_LON,
        "toplat": TOP_LAT,
        "bottomlat": BOTTOM_LAT,

        # GEFS directory
        "dir": f"/gefs.{DATE}/{CYCLE}/atmos/pgrb2sp25",
    }

    print()
    print("=" * 60)
    print(f"Downloading {member}")
    print(f"File: {filename}")
    print("=" * 60)

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=120,
    )

    print("HTTP status:", response.status_code)
    print("Downloaded:", len(response.content), "bytes")

    if response.status_code != 200:
        print("ERROR: NOAA request failed.")
        print(response.text[:500])
        return False

    # Make sure NOAA didn't return an HTML error page.
    content_type = response.headers.get("Content-Type", "")
    print("Content-Type:", content_type)

    if len(response.content) < 100:
        print("ERROR: Response is suspiciously small.")
        print(response.text[:500])
        return False

    output_file = OUTPUT_DIR / filename

    output_file.write_bytes(response.content)

    print("Saved:", output_file)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FORECASTMITR LIVE GEFS 5-MEMBER TEST")
    print("=" * 60)

    successful = []

    for member in MEMBERS:

        success = download_member(member)

        if success:
            successful.append(member)

        # Be polite to NOMADS.
        time.sleep(2)

    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print("Successful:", successful)
    print("Expected :", MEMBERS)

    if len(successful) == len(MEMBERS):
        print()
        print("SUCCESS: All 5 ensemble members downloaded.")
    else:
        print()
        print("WARNING: Some members failed.")


if __name__ == "__main__":
    main()