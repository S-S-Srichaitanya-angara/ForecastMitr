import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

GEFS_DIR = BASE_DIR / "data" / "gefs"


# ---------------------------------------------------------
# MVP dates selected from the GEFS archive
# ---------------------------------------------------------

DATES = [
    "2000010100",
    "2000011900",
    "2000020600",
    "2000022400",
    "2000031300",
    "2000033100",
    "2000041800",
    "2000050600",
    "2000052400",
    "2000061100",
    "2000062900",
    "2000071700",
    "2000080400",
    "2000082200",
    "2000090900",
    "2000092700",
    "2000101500",
    "2000110200",
    "2000112000",
    "2000120800",
]


MEMBERS = [
    "c00",
    "p01",
    "p02",
    "p03",
    "p04",
]


# ---------------------------------------------------------
# Download
# ---------------------------------------------------------

for date in DATES:

    print("\n" + "=" * 70)
    print(f"Processing {date}")
    print("=" * 70)

    year = date[:4]

    for member in MEMBERS:

        output_dir = (
            GEFS_DIR
            / date
            / member
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"apcp_sfc_{date}_{member}.grib2"
        )

        output_file = output_dir / filename

        # Skip files that already exist
        if output_file.exists():

            print(
                f"{member}: already exists → skipping"
            )

            continue

        s3_path = (
            f"s3://noaa-gefs-retrospective/"
            f"GEFSv12/reforecast/"
            f"{year}/{date}/{member}/"
            f"Days:1-10/{filename}"
        )

        print(f"{member}: downloading...")

        command = [
            "aws",
            "s3",
            "cp",
            s3_path,
            str(output_file),
            "--no-sign-request",
        ]

        result = subprocess.run(
            command
        )

        if result.returncode != 0:

            print(
                f"ERROR: failed to download "
                f"{date} {member}"
            )

        else:

            print(
                f"{member}: download complete"
            )


print("\n" + "=" * 70)
print("GEFS DOWNLOAD COMPLETE")
print("=" * 70)