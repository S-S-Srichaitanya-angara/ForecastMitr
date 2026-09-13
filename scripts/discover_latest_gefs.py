import re
import requests
from datetime import datetime


BASE_URL = (
    "https://nomads.ncep.noaa.gov/"
    "gribfilter.php?ds=gefs_atmos_0p25s"
)


def discover_latest():

    print()
    print("=" * 70)
    print("FORECASTMITR GEFS DISCOVERY")
    print("=" * 70)

    response = requests.get(
        BASE_URL,
        timeout=30
    )

    response.raise_for_status()

    html = response.text

    # Find available GEFS dates
    dates = sorted(
        set(
            re.findall(
                r"gefs\.(\d{8})",
                html
            )
        ),
        reverse=True
    )

    if not dates:
        raise RuntimeError(
            "Could not find available GEFS dates."
        )

    latest_date = dates[0]

    # Current page exposes cycles 00, 06, 12, 18.
    cycles = []

    for cycle in ["00", "06", "12", "18"]:

        if (
            f'value="{cycle}"' in html
            or f">{cycle}<" in html
        ):
            cycles.append(cycle)

    if not cycles:
        # Fall back to the standard operational cycles.
        cycles = ["00", "06", "12", "18"]

    latest_cycle = sorted(
        cycles,
        key=int,
        reverse=True
    )[0]

    issue_time = datetime.strptime(
        latest_date + latest_cycle,
        "%Y%m%d%H"
    )

    print()
    print(f"Latest available date : {latest_date}")
    print(f"Available cycles      : {cycles}")
    print(f"Selected cycle        : {latest_cycle}")
    print(f"Initialization        : {issue_time} UTC")

    print()
    print("=" * 70)


if __name__ == "__main__":
    discover_latest()