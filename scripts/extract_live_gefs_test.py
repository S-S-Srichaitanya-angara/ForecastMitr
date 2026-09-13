from pathlib import Path
import numpy as np

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_array,
    codes_get_values,
    codes_release,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/live_gefs_test")

TARGET_LAT = 13.0827
TARGET_LON = 80.2707

MEMBERS = {
    "c00": "gec00.t00z.pgrb2s.0p25.f003",
    "p01": "gep01.t00z.pgrb2s.0p25.f003",
    "p02": "gep02.t00z.pgrb2s.0p25.f003",
    "p03": "gep03.t00z.pgrb2s.0p25.f003",
    "p04": "gep04.t00z.pgrb2s.0p25.f003",
}


# ============================================================
# EXTRACT NEAREST CHENNAI GRID VALUE
# ============================================================

def extract_chennai_value(file_path):

    gid = None

    try:
        with open(file_path, "rb") as f:

            gid = codes_grib_new_from_file(f)

            if gid is None:
                raise RuntimeError(
                    f"Could not read GRIB message from {file_path}"
                )

            # ----------------------------------------------------
            # Read grid coordinates
            # ----------------------------------------------------

            lats = np.asarray(
                codes_get_array(gid, "latitudes"),
                dtype=float
            )

            lons = np.asarray(
                codes_get_array(gid, "longitudes"),
                dtype=float
            )

            # ----------------------------------------------------
            # Read precipitation values
            # ----------------------------------------------------

            values = np.asarray(
                codes_get_values(gid),
                dtype=float
            )

            # ----------------------------------------------------
            # Validate arrays
            # ----------------------------------------------------

            if len(lats) != len(lons):
                raise RuntimeError(
                    f"Latitude/longitude size mismatch: "
                    f"{len(lats)} vs {len(lons)}"
                )

            if len(lats) != len(values):
                raise RuntimeError(
                    f"Grid/value size mismatch: "
                    f"grid={len(lats)}, values={len(values)}"
                )

            if len(lats) == 0:
                raise RuntimeError(
                    "GRIB file contains no grid points"
                )

            # ----------------------------------------------------
            # Handle longitude convention
            # ----------------------------------------------------

            target_lon = TARGET_LON

            if np.nanmax(lons) > 180:
                target_lon = TARGET_LON % 360

            # ----------------------------------------------------
            # Find nearest grid point
            # ----------------------------------------------------

            distance = np.sqrt(
                (lats - TARGET_LAT) ** 2 +
                (lons - target_lon) ** 2
            )

            nearest_idx = np.nanargmin(distance)

            nearest_lat = lats[nearest_idx]
            nearest_lon = lons[nearest_idx]
            precipitation = values[nearest_idx]

            # ----------------------------------------------------
            # Metadata
            # ----------------------------------------------------

            short_name = codes_get(gid, "shortName")
            units = codes_get(gid, "units")
            step_type = codes_get(gid, "stepType")
            start_step = codes_get(gid, "startStep")
            end_step = codes_get(gid, "endStep")

            # ----------------------------------------------------
            # IMPORTANT: return "rainfall" explicitly
            # ----------------------------------------------------

            result = {
                "rainfall": float(precipitation),
                "latitude": float(nearest_lat),
                "longitude": float(nearest_lon),
                "distance": float(distance[nearest_idx]),
                "shortName": short_name,
                "units": units,
                "stepType": step_type,
                "startStep": start_step,
                "endStep": end_step,
            }

            return result

    finally:

        if gid is not None:
            codes_release(gid)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("FORECASTMITR LIVE GEFS ENSEMBLE EXTRACTION")
    print("=" * 65)

    rainfall_values = {}

    for member, filename in MEMBERS.items():

        file_path = DATA_DIR / filename

        print()
        print(f"Member: {member}")
        print(f"File  : {filename}")

        if not file_path.exists():

            print("ERROR: File not found.")
            continue

        result = extract_chennai_value(file_path)

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if "rainfall" not in result:
            raise RuntimeError(
                f"Extraction result does not contain 'rainfall'. "
                f"Returned keys: {list(result.keys())}"
            )

        rainfall_values[member] = result["rainfall"]

        print(
            f"Nearest grid : "
            f"{result['latitude']:.2f}, "
            f"{result['longitude']:.2f}"
        )

        print(
            f"Distance     : "
            f"{result['distance']:.4f} degrees"
        )

        print(
            f"Variable     : "
            f"{result['shortName']}"
        )

        print(
            f"Units        : "
            f"{result['units']}"
        )

        print(
            f"Accumulation : "
            f"{result['startStep']} -> "
            f"{result['endStep']} h"
        )

        print(
            f"Rainfall     : "
            f"{result['rainfall']:.4f} mm"
        )

    # ========================================================
    # CHECK ALL FIVE MEMBERS
    # ========================================================

    print()
    print("=" * 65)
    print("ENSEMBLE VALUES")
    print("=" * 65)

    if len(rainfall_values) != 5:

        print(
            f"ERROR: Expected 5 members, "
            f"found {len(rainfall_values)}."
        )

        return

    for member, value in rainfall_values.items():

        print(
            f"{member:5} : {value:.4f} mm"
        )

    values = np.array(
        list(rainfall_values.values()),
        dtype=float
    )

    # ========================================================
    # ENSEMBLE STATISTICS
    # ========================================================

    mean_value = np.mean(values)
    median_value = np.median(values)
    min_value = np.min(values)
    max_value = np.max(values)
    spread = max_value - min_value

    print()
    print("=" * 65)
    print("ENSEMBLE STATISTICS")
    print("=" * 65)

    print(f"Mean   : {mean_value:.4f} mm")
    print(f"Median : {median_value:.4f} mm")
    print(f"Min    : {min_value:.4f} mm")
    print(f"Max    : {max_value:.4f} mm")
    print(f"Spread : {spread:.4f} mm")

    print()
    print("=" * 65)
    print("SUCCESS")
    print("=" * 65)


if __name__ == "__main__":
    main()