import re
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://nomads.ncep.noaa.gov/"
    "pub/data/nccf/com/gens/prod/"
)

MEMBERS = [
    "c00",
    "p01",
    "p02",
    "p03",
    "p04",
]

FORECAST_HOUR = "003"


# ============================================================
# GET PRODUCTION DIRECTORY
# ============================================================

print("=" * 70)
print("FORECASTMITR — OPERATIONAL GEFS DISCOVERY")
print("=" * 70)

print("\nChecking NOAA/NOMADS production directory...")

response = requests.get(
    BASE_URL,
    timeout=30
)

response.raise_for_status()

html = response.text


# ============================================================
# FIND GEFS DIRECTORIES
# ============================================================

matches = re.findall(
    r'href="gefs\.(\d{10})/',
    html
)

dates = sorted(
    set(matches),
    reverse=True
)

if not dates:
    raise RuntimeError(
        "No GEFS production directories found."
    )


print(
    f"GEFS runs discovered: {len(dates)}"
)

print("\nMost recent runs:")
for run in dates[:10]:
    print(f"  {run}")


# ============================================================
# TEST ACTUAL FILE AVAILABILITY
# ============================================================

def file_exists(date_time, member):

    date = date_time[:8]
    cycle = date_time[8:10]

    if member == "c00":
        prefix = "gec00"
    else:
        prefix = f"gep{member[1:]}"

    filename = (
        f"{prefix}.t{cycle}z."
        f"pgrb2s.0p25.f{FORECAST_HOUR}"
    )

    url = (
        "https://nomads.ncep.noaa.gov/"
        "cgi-bin/filter_gefs_atmos_0p25s.pl"
        f"?file={filename}"
        f"&dir=%2Fgefs.{date}{cycle}%2Fatmos"
        "&var_APCP=on"
        "&lev_surface=on"
        "&leftlon=79.75"
        "&rightlon=80.75"
        "&toplat=13.75"
        "&bottomlat=12.75"
    )

    try:
        r = requests.get(
            url,
            timeout=30
        )

        return (
            r.status_code == 200
            and len(r.content) > 100
        )

    except requests.RequestException:
        return False


# ============================================================
# FIND MOST RECENT COMPLETE RUN
# ============================================================

print("\nTesting actual forecast availability...")

selected_run = None

for run in dates:

    print(
        f"\nChecking {run[:8]} "
        f"{run[8:10]} UTC..."
    )

    available = []

    for member in MEMBERS:

        ok = file_exists(
            run,
            member
        )

        status = "OK" if ok else "NO"

        print(
            f"  {member}: {status}"
        )

        if ok:
            available.append(member)

    if len(available) == len(MEMBERS):

        selected_run = run

        print(
            "\nCOMPLETE RUN FOUND"
        )

        break


# ============================================================
# RESULT
# ============================================================

if selected_run is None:

    raise RuntimeError(
        "No complete GEFS run found for "
        "c00 + p01–p04."
    )


print("\n" + "=" * 70)
print("DISCOVERY SUCCESS")
print("=" * 70)

print(
    f"Selected initialization : "
    f"{selected_run[:8]} "
    f"{selected_run[8:10]} UTC"
)

print(
    f"Forecast hour           : "
    f"f{FORECAST_HOUR}"
)

print(
    "Members                 : "
    + ", ".join(MEMBERS)
)

print("=" * 70)