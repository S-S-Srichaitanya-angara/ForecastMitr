from pathlib import Path
from datetime import datetime, timedelta
import sys
import math

import eccodes


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "live_gefs_latest"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from forecastmitr_engine import predict


# ============================================================
# CHENNAI
# ============================================================

CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707


# ============================================================
# ENSEMBLE MEMBERS
# ============================================================

MEMBERS = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]


# ============================================================
# FIND LATEST COMPLETE GEFS CYCLE
# ============================================================

def find_latest_cycle():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"GEFS directory not found: {DATA_DIR}"
        )

    candidates = []

    for directory in DATA_DIR.iterdir():

        if not directory.is_dir():
            continue

        complete = True

        for member in MEMBERS:

            matches = list(
                directory.glob(
                    f"{member}.t*.pgrb2s.0p25.f*"
                )
            )

            if not matches:
                complete = False
                break

        if complete:
            candidates.append(directory)

    if not candidates:
        raise RuntimeError(
            "No complete GEFS cycle found."
        )

    candidates.sort(
        key=lambda p: p.name,
        reverse=True
    )

    return candidates[0]


# ============================================================
# FIND MEMBER FILE
# ============================================================

def find_member_file(cycle_dir, member):

    matches = list(
        cycle_dir.glob(
            f"{member}.t*.pgrb2s.0p25.f*"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No GEFS file found for {member} "
            f"in {cycle_dir}"
        )

    return matches[0]


# ============================================================
# READ GEFS PRECIPITATION
# ============================================================

def read_grib_precipitation(filepath):

    with open(filepath, "rb") as f:

        while True:

            gid = eccodes.codes_grib_new_from_file(f)

            if gid is None:
                break

            try:

                short_name = eccodes.codes_get(
                    gid,
                    "shortName"
                )

                # We only want total precipitation.
                if short_name != "tp":
                    continue

                lats = eccodes.codes_get_array(
                    gid,
                    "latitudes"
                )

                lons = eccodes.codes_get_array(
                    gid,
                    "longitudes"
                )

                values = eccodes.codes_get_array(
                    gid,
                    "values"
                )

                best_index = None
                best_distance = float("inf")

                for i, (lat, lon) in enumerate(
                    zip(lats, lons)
                ):

                    # Convert 0–360 longitude to -180–180.
                    if lon > 180:
                        lon -= 360

                    distance = (
                        (float(lat) - CHENNAI_LAT) ** 2
                        +
                        (float(lon) - CHENNAI_LON) ** 2
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_index = i

                if best_index is None:
                    raise RuntimeError(
                        "Could not find nearest grid point."
                    )

                rainfall_mm = float(
                    values[best_index]
                )

                nearest_lat = float(
                    lats[best_index]
                )

                nearest_lon = float(
                    lons[best_index]
                )

                start_step = int(
                    eccodes.codes_get(
                        gid,
                        "startStep"
                    )
                )

                end_step = int(
                    eccodes.codes_get(
                        gid,
                        "endStep"
                    )
                )

                return {
                    "rainfall_mm": rainfall_mm,
                    "lat": nearest_lat,
                    "lon": nearest_lon,
                    "distance": math.sqrt(
                        best_distance
                    ),
                    "start_step": start_step,
                    "end_step": end_step,
                }

            finally:

                eccodes.codes_release(gid)

    raise RuntimeError(
        f"No total precipitation field found in:\n"
        f"{filepath}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FORECASTMITR - AUTOMATIC LIVE INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Locate latest COMPLETE cycle
    # --------------------------------------------------------

    cycle_dir = find_latest_cycle()

    print()
    print(
        f"GEFS directory : {cycle_dir}"
    )

    cycle_name = cycle_dir.name

    try:

        date_str, cycle_str = cycle_name.split("_")

        issue_time = datetime.strptime(
            date_str + cycle_str,
            "%Y%m%d%H"
        )

    except Exception:

        raise RuntimeError(
            f"Unexpected cycle directory name: "
            f"{cycle_name}"
        )

    print(
        f"Initialization : "
        f"{issue_time:%Y-%m-%d %H:%M} UTC"
    )

    # --------------------------------------------------------
    # Extract members
    # --------------------------------------------------------

    extracted = {}

    print()
    print("-" * 70)
    print("EXTRACTING CHENNAI PRECIPITATION")
    print("-" * 70)

    for member in MEMBERS:

        filepath = find_member_file(
            cycle_dir,
            member
        )

        result = read_grib_precipitation(
            filepath
        )

        extracted[member] = result

        print(
            f"{member:5s} : "
            f"{result['rainfall_mm']:.3f} mm"
        )

    # --------------------------------------------------------
    # Verify same forecast lead
    # --------------------------------------------------------

    end_steps = {
        extracted[m]["end_step"]
        for m in MEMBERS
    }

    if len(end_steps) != 1:

        raise RuntimeError(
            "Ensemble members have different "
            "forecast lead times."
        )

    lead_hours = end_steps.pop()

    target_time = (
        issue_time
        +
        timedelta(hours=lead_hours)
    )

    # --------------------------------------------------------
    # Ensemble statistics
    #
    # IMPORTANT:
    # c00 = deterministic/control forecast
    # p01-p04 = perturbed ensemble members
    # --------------------------------------------------------

    control_forecast = (
        extracted["gec00"]["rainfall_mm"]
    )

    perturbed = [
        extracted[m]["rainfall_mm"]
        for m in MEMBERS
        if m != "gec00"
    ]

    ensemble_mean = (
        sum(perturbed)
        /
        len(perturbed)
    )

    sorted_values = sorted(perturbed)

    n = len(sorted_values)

    if n % 2 == 1:

        ensemble_median = (
            sorted_values[n // 2]
        )

    else:

        ensemble_median = (
            sorted_values[n // 2 - 1]
            +
            sorted_values[n // 2]
        ) / 2

    ensemble_min = min(perturbed)

    ensemble_max = max(perturbed)

    ensemble_spread = (
        ensemble_max
        -
        ensemble_min
    )

    # --------------------------------------------------------
    # Display statistics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("ENSEMBLE STATISTICS")
    print("-" * 70)

    print(
        f"Control forecast : "
        f"{control_forecast:.3f} mm"
    )

    print(
        f"Ensemble mean   : "
        f"{ensemble_mean:.3f} mm"
    )

    print(
        f"Ensemble median : "
        f"{ensemble_median:.3f} mm"
    )

    print(
        f"Ensemble min    : "
        f"{ensemble_min:.3f} mm"
    )

    print(
        f"Ensemble max    : "
        f"{ensemble_max:.3f} mm"
    )

    print(
        f"Ensemble spread : "
        f"{ensemble_spread:.3f} mm"
    )

    # --------------------------------------------------------
    # ForecastMitr
    # --------------------------------------------------------

    prediction = predict(
        control_forecast=control_forecast,
        ensemble_mean=ensemble_mean,
        ensemble_median=ensemble_median,
        ensemble_min=ensemble_min,
        ensemble_max=ensemble_max,
        forecast_time_hours=lead_hours,
        month=issue_time.month,
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FORECASTMITR RESULT")
    print("=" * 70)

    print(
        f"Issue time      : "
        f"{issue_time:%Y-%m-%d %H:%M} UTC"
    )

    print(
        f"Target time     : "
        f"{target_time:%Y-%m-%d %H:%M} UTC"
    )

    print(
        f"Lead time       : "
        f"+{lead_hours} hours"
    )

    print(
        f"Chennai grid    : "
        f"{extracted['gec00']['lat']:.2f}, "
        f"{extracted['gec00']['lon']:.2f}"
    )

    print(
        f"Grid distance   : "
        f"{extracted['gec00']['distance']:.4f}°"
    )

    print()

    print(
        f"Bust probability: "
        f"{prediction['bust_probability'] * 100:.1f}%"
    )

    print(
        f"Predicted bust  : "
        f"{'YES' if prediction['predicted_bust'] else 'NO'}"
    )

    print(
        f"Risk level      : "
        f"{prediction['risk_level']}"
    )

    print(
        f"Diagnosis       : "
        f"{prediction['diagnosis']}"
    )

    print(
        f"Primary signal  : "
        f"{prediction['primary_signal']}"
    )

    if prediction.get("signals"):

        print()
        print("Signals:")

        for signal in prediction["signals"]:

            print(
                f"  • {signal}"
            )

    if prediction.get("recommendation"):

        print()
        print(
            f"Recommendation  : "
            f"{prediction['recommendation']}"
        )

    print()
    print("=" * 70)
    print("AUTOMATIC LIVE INFERENCE SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()