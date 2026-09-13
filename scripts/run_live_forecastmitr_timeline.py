from pathlib import Path
from datetime import datetime, timedelta
import sys
import math
import csv

import eccodes


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "live_gefs_timeline"

OUTPUT_FILE = (
    PROJECT_ROOT /
    "data" /
    "forecastmitr_live_timeline.csv"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts")
)

from forecastmitr_engine import predict


# ============================================================
# CHENNAI
# ============================================================

CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707


# ============================================================
# MEMBERS
# ============================================================

MEMBERS = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]


# ============================================================
# FIND LATEST COMPLETE CYCLE
# ============================================================

def find_latest_cycle():

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Directory not found:\n{DATA_DIR}"
        )

    candidates = []

    for directory in DATA_DIR.iterdir():

        if not directory.is_dir():
            continue

        complete = True

        for member in MEMBERS:

            files = list(
                directory.glob(
                    f"{member}.t*.pgrb2s.0p25.f*"
                )
            )

            if not files:

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
# FIND FILE
# ============================================================

def find_member_file(
    cycle_dir,
    member,
    forecast_hour
):

    filename = (
        f"{member}.t*.pgrb2s.0p25."
        f"f{forecast_hour:03d}"
    )

    matches = list(
        cycle_dir.glob(filename)
    )

    if not matches:

        raise FileNotFoundError(
            f"Missing {member} "
            f"+{forecast_hour}h"
        )

    return matches[0]


# ============================================================
# READ PRECIPITATION
# ============================================================

def read_grib_precipitation(filepath):

    with open(filepath, "rb") as f:

        while True:

            gid = (
                eccodes.codes_grib_new_from_file(f)
            )

            if gid is None:
                break

            try:

                short_name = eccodes.codes_get(
                    gid,
                    "shortName"
                )

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

                    lat = float(lat)
                    lon = float(lon)

                    if lon > 180:
                        lon -= 360

                    distance = (
                        (lat - CHENNAI_LAT) ** 2
                        +
                        (lon - CHENNAI_LON) ** 2
                    )

                    if distance < best_distance:

                        best_distance = distance
                        best_index = i

                if best_index is None:

                    raise RuntimeError(
                        "Nearest Chennai grid "
                        "point could not be found."
                    )

                rainfall_mm = float(
                    values[best_index]
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
                    "lat": float(
                        lats[best_index]
                    ),
                    "lon": float(
                        lons[best_index]
                    ),
                    "distance": math.sqrt(
                        best_distance
                    ),
                    "start_step": start_step,
                    "end_step": end_step,
                }

            finally:

                eccodes.codes_release(gid)

    raise RuntimeError(
        f"No total precipitation field found:\n"
        f"{filepath}"
    )


# ============================================================
# MEDIAN
# ============================================================

def calculate_median(values):

    values = sorted(values)

    n = len(values)

    if n == 0:
        return 0.0

    if n % 2 == 1:

        return values[n // 2]

    return (
        values[n // 2 - 1]
        +
        values[n // 2]
    ) / 2.0


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "FORECASTMITR - LIVE BUST TIMELINE"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Locate cycle
    # --------------------------------------------------------

    cycle_dir = find_latest_cycle()

    print()
    print(
        f"GEFS cycle directory:"
    )
    print(cycle_dir)

    date_str, cycle_str = (
        cycle_dir.name.split("_")
    )

    issue_time = datetime.strptime(
        date_str + cycle_str,
        "%Y%m%d%H"
    )

    print(
        f"Initialization : "
        f"{issue_time:%Y-%m-%d %H:%M} UTC"
    )

    # --------------------------------------------------------
    # Discover available forecast hours
    # --------------------------------------------------------

    available_hours = set()

    for filepath in cycle_dir.glob(
        "gec00.t*.pgrb2s.0p25.f*"
    ):

        name = filepath.name

        try:

            forecast_part = name.split(
                ".f"
            )[-1]

            forecast_hour = int(
                forecast_part
            )

            available_hours.add(
                forecast_hour
            )

        except ValueError:

            continue

    forecast_hours = sorted(
        available_hours
    )

    if not forecast_hours:

        raise RuntimeError(
            "No forecast-hour files found."
        )

    print(
        f"Forecast windows: "
        f"{len(forecast_hours)}"
    )

    print(
        f"Range: "
        f"+{forecast_hours[0]}h "
        f"to "
        f"+{forecast_hours[-1]}h"
    )

    # --------------------------------------------------------
    # Process every lead
    # --------------------------------------------------------

    rows = []

    print()
    print("-" * 78)
    print(
        "FORECASTMITR LIVE TIMELINE"
    )
    print("-" * 78)

    print(
        f"{'LEAD':>6} "
        f"{'CTRL':>8} "
        f"{'MEAN':>8} "
        f"{'SPREAD':>8} "
        f"{'BUST':>8} "
        f"{'RISK':>10} "
        f"DIAGNOSIS"
    )

    print("-" * 78)

    for forecast_hour in forecast_hours:

        extracted = {}

        # ----------------------------------------------------
        # Read all ensemble members
        # ----------------------------------------------------

        for member in MEMBERS:

            filepath = find_member_file(
                cycle_dir,
                member,
                forecast_hour
            )

            extracted[member] = (
                read_grib_precipitation(
                    filepath
                )
            )

        # ----------------------------------------------------
        # Verify lead time
        # ----------------------------------------------------

        actual_end_steps = {
            extracted[m]["end_step"]
            for m in MEMBERS
        }

        if len(actual_end_steps) != 1:

            raise RuntimeError(
                f"Different endStep values "
                f"at +{forecast_hour}h"
            )

        lead_hours = (
            actual_end_steps.pop()
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        control_forecast = (
            extracted["gec00"]
            ["rainfall_mm"]
        )

        ensemble_values = [
            extracted[m]["rainfall_mm"]
            for m in MEMBERS
            if m != "gec00"
        ]

        ensemble_mean = (
            sum(ensemble_values)
            /
            len(ensemble_values)
        )

        ensemble_median = (
            calculate_median(
                ensemble_values
            )
        )

        ensemble_min = min(
            ensemble_values
        )

        ensemble_max = max(
            ensemble_values
        )

        ensemble_spread = (
            ensemble_max
            -
            ensemble_min
        )

        # ----------------------------------------------------
        # ForecastMitr
        # ----------------------------------------------------

        prediction = predict(
            control_forecast=control_forecast,
            ensemble_mean=ensemble_mean,
            ensemble_median=ensemble_median,
            ensemble_min=ensemble_min,
            ensemble_max=ensemble_max,
            forecast_time_hours=lead_hours,
            month=issue_time.month,
        )

        probability = (
            prediction[
                "bust_probability"
            ]
        )

        predicted_bust = (
            prediction[
                "predicted_bust"
            ]
        )

        risk_level = (
            prediction[
                "risk_level"
            ]
        )

        diagnosis = (
            prediction[
                "diagnosis"
            ]
        )

        primary_signal = (
            prediction[
                "primary_signal"
            ]
        )

        target_time = (
            issue_time
            +
            timedelta(
                hours=lead_hours
            )
        )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        rows.append({

            "issue_time":
                issue_time.isoformat(),

            "target_time":
                target_time.isoformat(),

            "lead_hours":
                lead_hours,

            "control_forecast":
                control_forecast,

            "ensemble_mean":
                ensemble_mean,

            "ensemble_median":
                ensemble_median,

            "ensemble_min":
                ensemble_min,

            "ensemble_max":
                ensemble_max,

            "ensemble_spread":
                ensemble_spread,

            "bust_probability":
                probability,

            "predicted_bust":
                predicted_bust,

            "risk_level":
                risk_level,

            "diagnosis":
                diagnosis,

            "primary_signal":
                primary_signal,
        })

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print(
            f"{lead_hours:>5}h "
            f"{control_forecast:>8.2f} "
            f"{ensemble_mean:>8.2f} "
            f"{ensemble_spread:>8.2f} "
            f"{probability * 100:>7.1f}% "
            f"{risk_level:>10} "
            f"{diagnosis}"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    bust_rows = [
        row
        for row in rows
        if row["predicted_bust"]
    ]

    high_rows = [
        row
        for row in rows
        if row["risk_level"]
        in ["HIGH", "CRITICAL"]
    ]

    print()
    print("=" * 78)
    print("LIVE BUST TIMELINE SUMMARY")
    print("=" * 78)

    print(
        f"Total forecast windows : "
        f"{len(rows)}"
    )

    print(
        f"Predicted bust windows : "
        f"{len(bust_rows)}"
    )

    print(
        f"High/Critical windows  : "
        f"{len(high_rows)}"
    )

    if bust_rows:

        earliest = min(
            bust_rows,
            key=lambda x:
            x["lead_hours"]
        )

        print()
        print(
            "EARLIEST PREDICTED BUST:"
        )

        print(
            f"  Lead       : "
            f"+{earliest['lead_hours']} h"
        )

        print(
            f"  Probability: "
            f"{earliest['bust_probability'] * 100:.1f}%"
        )

        print(
            f"  Risk       : "
            f"{earliest['risk_level']}"
        )

        print(
            f"  Diagnosis  : "
            f"{earliest['diagnosis']}"
        )

    else:

        print()
        print(
            "No predicted bust window "
            "in the available forecast horizon."
        )

    print()
    print(
        f"Saved timeline:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 78)
    print(
        "FORECASTMITR LIVE TIMELINE SUCCESS"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()