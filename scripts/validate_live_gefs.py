from pathlib import Path
from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
)


TEST_DIR = Path("data/live_gefs_test")


def main():
    files = list(TEST_DIR.glob("*"))

    if not files:
        print("ERROR: No downloaded file found.")
        print(f"Put the GRIB2 file inside: {TEST_DIR}")
        return

    file_path = files[0]

    print("=" * 60)
    print("FORECASTMITR LIVE GEFS VALIDATION")
    print("=" * 60)
    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
    print()

    with open(file_path, "rb") as f:
        message_number = 0

        while True:
            gid = codes_grib_new_from_file(f)

            if gid is None:
                break

            message_number += 1

            print(f"GRIB MESSAGE {message_number}")
            print("-" * 40)

            keys = [
                "shortName",
                "name",
                "units",
                "stepType",
                "startStep",
                "endStep",
                "forecastTime",
                "Ni",
                "Nj",
                "latitudeOfFirstGridPointInDegrees",
                "longitudeOfFirstGridPointInDegrees",
                "latitudeOfLastGridPointInDegrees",
                "longitudeOfLastGridPointInDegrees",
            ]

            for key in keys:
                try:
                    value = codes_get(gid, key)
                    print(f"{key:45} {value}")
                except Exception:
                    print(f"{key:45} <not available>")

            codes_release(gid)

    print()
    print("=" * 60)
    print(f"TOTAL GRIB MESSAGES: {message_number}")
    print("=" * 60)


if __name__ == "__main__":
    main()