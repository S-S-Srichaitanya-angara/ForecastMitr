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
    """Read the complete GEFS precipitation grid from a GRIB file.

    Returns the grid values plus the matching latitude/longitude arrays.
    The existing timeline only needs the nearest Chennai point, but the
    complete grid is also retained so the dashboard can render a real
    geographic Chennai heatmap without inventing spatial values.
    """

    with open(filepath, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break

            try:
                short_name = eccodes.codes_get(gid, "shortName")

                if short_name != "tp":
                    continue

                lats = [float(v) for v in eccodes.codes_get_array(gid, "latitudes")]
                lons = [float(v) for v in eccodes.codes_get_array(gid, "longitudes")]
                values = [float(v) for v in eccodes.codes_get_array(gid, "values")]

                normalized_lons = []
                for lon in lons:
                    if lon > 180:
                        lon -= 360
                    normalized_lons.append(lon)

                start_step = int(eccodes.codes_get(gid, "startStep"))
                end_step = int(eccodes.codes_get(gid, "endStep"))

                return {
                    "rainfall_mm": values,
                    "latitudes": lats,
                    "longitudes": normalized_lons,
                    "start_step": start_step,
                    "end_step": end_step,
                }

            finally:
                eccodes.codes_release(gid)

    raise RuntimeError(
        f"No total precipitation field found:\n{filepath}"
    )


def extract_chennai_point(grid):
    """Return the nearest grid point to the Chennai reference location."""

    best_index = None
    best_distance = float("inf")

    for i, (lat, lon) in enumerate(
        zip(grid["latitudes"], grid["longitudes"])
    ):
        distance = (
            (lat - CHENNAI_LAT) ** 2
            + (lon - CHENNAI_LON) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_index = i

    if best_index is None:
        raise RuntimeError("Nearest Chennai grid point could not be found.")

    return {
        "rainfall_mm": grid["rainfall_mm"][best_index],
        "lat": grid["latitudes"][best_index],
        "lon": grid["longitudes"][best_index],
        "distance": math.sqrt(best_distance),
        "start_step": grid["start_step"],
        "end_step": grid["end_step"],
    }


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
    spatial_rows = []

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

            extracted[member] = read_grib_precipitation(filepath)

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

        point_extracted = {
            member: extract_chennai_point(extracted[member])
            for member in MEMBERS
        }

        control_forecast = point_extracted["gec00"]["rainfall_mm"]

        # FIX: ensemble statistics must include every member,
        # including the control run (gec00) — this matches how
        # calculate_ensemble_stats.py builds the TRAINING data
        # (it groups over all 5 members, control included).
        # Excluding the control here previously created a
        # train/serve mismatch: the model was trained on 5-member
        # ensemble stats but scored live on 4-member stats.
        ensemble_values = [
            point_extracted[m]["rainfall_mm"]
            for m in MEMBERS
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
        # Save complete spatial ensemble grid for the dashboard
        # ----------------------------------------------------
        reference_grid = extracted[MEMBERS[0]]
        grid_size = len(reference_grid["rainfall_mm"])

        for member in MEMBERS[1:]:
            member_grid = extracted[member]
            if len(member_grid["rainfall_mm"]) != grid_size:
                raise RuntimeError(
                    f"Grid size mismatch for {member} at +{forecast_hour}h"
                )
            if member_grid["latitudes"] != reference_grid["latitudes"] or \
               member_grid["longitudes"] != reference_grid["longitudes"]:
                raise RuntimeError(
                    f"Grid coordinates differ for {member} at +{forecast_hour}h"
                )

        for grid_index in range(grid_size):
            grid_values = [
                extracted[member]["rainfall_mm"][grid_index]
                for member in MEMBERS
            ]

            grid_control = extracted["gec00"]["rainfall_mm"][grid_index]
            grid_mean = sum(grid_values) / len(grid_values)
            grid_median = calculate_median(grid_values)
            grid_min = min(grid_values)
            grid_max = max(grid_values)
            grid_spread = grid_max - grid_min

            # Spatial uncertainty is derived from the actual GEFS ensemble,
            # not from the single Chennai XGBoost probability.  The XGBoost
            # model has no latitude/longitude features, so its probability
            # is kept as a Chennai point forecast only.
            relative_spread = (
                grid_spread / (abs(grid_mean) + 0.10)
            )
            control_divergence = (
                abs(grid_control - grid_mean) /
                (abs(grid_mean) + 0.10)
            )
            spatial_uncertainty_raw = (
                0.70 * relative_spread +
                0.30 * control_divergence
            )

            spatial_rows.append({
                "issue_time": issue_time.isoformat(),
                "target_time": (
                    issue_time + timedelta(hours=lead_hours)
                ).isoformat(),
                "lead_hours": lead_hours,
                "lat": reference_grid["latitudes"][grid_index],
                "lon": reference_grid["longitudes"][grid_index],
                "control_forecast": grid_control,
                "ensemble_mean": grid_mean,
                "ensemble_median": grid_median,
                "ensemble_min": grid_min,
                "ensemble_max": grid_max,
                "ensemble_spread": grid_spread,
                "relative_spread": relative_spread,
                "control_divergence": control_divergence,
                "spatial_uncertainty_raw": spatial_uncertainty_raw,
            })

        # Convert the raw spatial disagreement into a 0-100 percentile
        # within this forecast lead. This is an uncertainty INDEX, not a
        # calibrated weather-event probability. It is guaranteed to use the
        # actual spatial distribution of the GEFS ensemble.
        lead_spatial = [
            row for row in spatial_rows
            if row["lead_hours"] == lead_hours
        ]
        raw_values = [
            float(row["spatial_uncertainty_raw"])
            for row in lead_spatial
        ]
        ordered = sorted(raw_values)
        rank_values = {}
        for raw in ordered:
            rank_values[raw] = (
                100.0 * ordered.index(raw) / max(1, len(ordered) - 1)
            )
        for row in lead_spatial:
            row["spatial_uncertainty_index"] = rank_values[
                float(row["spatial_uncertainty_raw"])
            ]

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
    # SAVE SPATIAL CHENNAI GRID
    # ========================================================

    spatial_output = (
        OUTPUT_FILE.parent /
        "forecastmitr_chennai_grid.csv"
    )

    spatial_fieldnames = list(spatial_rows[0].keys())

    with open(
        spatial_output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=spatial_fieldnames
        )
        writer.writeheader()
        writer.writerows(spatial_rows)

    print()
    print("Saved Chennai spatial grid:")
    print(spatial_output)

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