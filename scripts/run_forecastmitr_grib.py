from pathlib import Path
import sys
import numpy as np
import pandas as pd
import eccodes

sys.path.append(str(Path(__file__).resolve().parent))

from forecastmitr_engine import predict


# ============================================================
# CONFIGURATION
# ============================================================

GEFS_ROOT = Path("data/gefs")

MEMBERS = ["c00", "p01", "p02", "p03", "p04"]

TARGET_LAT = 13.0827
TARGET_LON = 80.2707


# ============================================================
# FIND GRIB FILE
# ============================================================

def find_grib_file(issue_dir, member):
    member_dir = issue_dir / member

    files = list(member_dir.rglob("*.grib2"))

    if not files:
        raise FileNotFoundError(
            f"No GRIB2 file found for {member} in {member_dir}"
        )

    return files[0]


# ============================================================
# FIND NEAREST GRID POINT
# ============================================================

def find_nearest_index(lats, lons, target_lat, target_lon):

    distance = (
        (lats - target_lat) ** 2
        + (lons - target_lon) ** 2
    )

    return np.unravel_index(
        np.argmin(distance),
        distance.shape
    )


# ============================================================
# READ 3-HOUR PRECIPITATION
# ============================================================

def extract_member_data(grib_file):

    records = []

    with open(grib_file, "rb") as f:

        while True:

            try:
                gid = eccodes.codes_grib_new_from_file(f)
            except Exception:
                break

            if gid is None:
                break

            try:

                short_name = eccodes.codes_get(
                    gid, "shortName"
                )

                if short_name != "tp":
                    continue

                start_step = eccodes.codes_get(
                    gid, "startStep"
                )

                end_step = eccodes.codes_get(
                    gid, "endStep"
                )

                # Only 3-hour accumulation
                if end_step - start_step != 3:
                    continue

                forecast_time = eccodes.codes_get(
                    gid, "forecastTime"
                )

                values = np.array(
                    eccodes.codes_get_array(
                        gid, "values"
                    )
                )

                ni = eccodes.codes_get(
                    gid, "Ni"
                )

                nj = eccodes.codes_get(
                    gid, "Nj"
                )

                lats = np.array(
                    eccodes.codes_get_array(
                        gid, "latitudes"
                    )
                ).reshape(nj, ni)

                lons = np.array(
                    eccodes.codes_get_array(
                        gid, "longitudes"
                    )
                ).reshape(nj, ni)

                row, col = find_nearest_index(
                    lats,
                    lons,
                    TARGET_LAT,
                    TARGET_LON
                )

                point_value = float(
                    values.reshape(nj, ni)[row, col]
                )

                # GEFS total precipitation is kg/m²,
                # numerically equivalent to mm of water.
                records.append(
                    {
                        "forecast_time_hours": float(
                            forecast_time
                        ),
                        "start_step_hours": float(
                            start_step
                        ),
                        "end_step_hours": float(
                            end_step
                        ),
                        "forecast_rainfall_mm": point_value,
                    }
                )

            finally:
                eccodes.codes_release(gid)

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    if not GEFS_ROOT.exists():
        raise FileNotFoundError(
            f"GEFS directory not found: {GEFS_ROOT}"
        )

    issue_dirs = sorted(
        [
            p for p in GEFS_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    if not issue_dirs:
        raise ValueError(
            "No GEFS initialization directories found."
        )

    # Use the first existing initialization for this dry run.
    issue_dir = issue_dirs[0]

    issue_time = pd.to_datetime(
        issue_dir.name,
        format="%Y%m%d%H"
    )

    print()
    print("=" * 70)
    print("FORECASTMITR DIRECT GRIB2 INFERENCE")
    print("=" * 70)

    print(f"Initialization : {issue_time}")
    print(f"Location       : Chennai")
    print(
        f"Nearest target : "
        f"{TARGET_LAT}, {TARGET_LON}"
    )

    # --------------------------------------------------------
    # Extract every ensemble member
    # --------------------------------------------------------

    member_tables = []

    for member in MEMBERS:

        grib_file = find_grib_file(
            issue_dir,
            member
        )

        print()
        print(
            f"Reading {member}: "
            f"{grib_file}"
        )

        member_df = extract_member_data(
            grib_file
        )

        member_df["ensemble_member"] = member

        print(
            f"  3-hour records: "
            f"{len(member_df)}"
        )

        member_tables.append(member_df)

    ensemble = pd.concat(
        member_tables,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Make sure every forecast lead has all 5 members
    # --------------------------------------------------------

    counts = (
        ensemble
        .groupby("forecast_time_hours")
        ["ensemble_member"]
        .nunique()
    )

    valid_leads = counts[
        counts == len(MEMBERS)
    ].index

    ensemble = ensemble[
        ensemble["forecast_time_hours"].isin(
            valid_leads
        )
    ].copy()

    if ensemble.empty:
        raise ValueError(
            "No forecast lead has all five ensemble members."
        )

    # --------------------------------------------------------
    # Select a future forecast lead for demonstration
    # --------------------------------------------------------

    lead_time = sorted(
        ensemble["forecast_time_hours"]
        .unique()
    )[1]

    case = ensemble[
        ensemble["forecast_time_hours"]
        == lead_time
    ].copy()

    # --------------------------------------------------------
    # Calculate ensemble statistics
    # --------------------------------------------------------

    values = {}

    for member in MEMBERS:

        row = case[
            case["ensemble_member"]
            == member
        ]

        if len(row) != 1:
            raise ValueError(
                f"Expected one row for {member}, "
                f"found {len(row)}"
            )

        values[member] = float(
            row["forecast_rainfall_mm"].iloc[0]
        )

    rainfall = list(values.values())

    ensemble_mean = float(
        np.mean(rainfall)
    )

    ensemble_median = float(
        np.median(rainfall)
    )

    ensemble_min = float(
        np.min(rainfall)
    )

    ensemble_max = float(
        np.max(rainfall)
    )

    ensemble_spread = (
        ensemble_max - ensemble_min
    )

    # --------------------------------------------------------
    # Target time
    # --------------------------------------------------------

    # For the 3-hour accumulation,
    # the target is the end of the accumulation window.
    target_time = (
        issue_time
        + pd.Timedelta(
            hours=float(
                case["end_step_hours"].iloc[0]
            )
        )
    )

    month = int(target_time.month)

    # --------------------------------------------------------
    # ForecastMitr
    # --------------------------------------------------------

    result = predict(
        control_forecast=values["c00"],
        ensemble_mean=ensemble_mean,
        ensemble_median=ensemble_median,
        ensemble_min=ensemble_min,
        ensemble_max=ensemble_max,
        forecast_time_hours=lead_time,
        month=month,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GEFS FORECAST")
    print("=" * 70)

    print(f"Issue time     : {issue_time}")
    print(f"Target time    : {target_time}")
    print(f"Lead time      : {lead_time:.0f} hours")

    print()
    print("ENSEMBLE")
    print("-" * 70)

    for member in MEMBERS:
        print(
            f"{member:15}: "
            f"{values[member]:.3f} mm/3h"
        )

    print(
        f"{'Mean':15}: "
        f"{ensemble_mean:.3f} mm/3h"
    )

    print(
        f"{'Median':15}: "
        f"{ensemble_median:.3f} mm/3h"
    )

    print(
        f"{'Min':15}: "
        f"{ensemble_min:.3f} mm/3h"
    )

    print(
        f"{'Max':15}: "
        f"{ensemble_max:.3f} mm/3h"
    )

    print(
        f"{'Spread':15}: "
        f"{ensemble_spread:.3f} mm"
    )

    print()
    print("=" * 70)
    print("FORECASTMITR")
    print("=" * 70)

    print(
        f"Bust probability : "
        f"{result['bust_probability']:.1%}"
    )

    print(
        f"Risk level       : "
        f"{result['risk_level']}"
    )

    print(
        f"Bust detected    : "
        f"{'YES' if result['predicted_bust'] else 'NO'}"
    )

    print(
        f"Diagnosis        : "
        f"{result['diagnosis']}"
    )

    print()
    print(
        f"Primary signal   : "
        f"{result['primary_signal']}"
    )

    print()
    print("Signals:")

    for signal in result["signals"]:
        print(f"  - {signal}")

    print()
    print("Recommendation:")
    print(f"  {result['recommendation']}")

    print("=" * 70)


if __name__ == "__main__":
    main()