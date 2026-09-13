import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


YEAR = "2000"


print("=" * 70)
print("GEFS HISTORICAL INITIALIZATION DATES")
print("=" * 70)

command = [
    "aws",
    "s3",
    "ls",
    f"s3://noaa-gefs-retrospective/GEFSv12/reforecast/{YEAR}/",
    "--no-sign-request",
]


result = subprocess.run(
    command,
    capture_output=True,
    text=True,
)


if result.returncode != 0:
    print("AWS S3 listing failed.")
    print(result.stderr)
    raise SystemExit(1)


dates = []

for line in result.stdout.splitlines():

    parts = line.split()

    if not parts:
        continue

    name = parts[-1].rstrip("/")

    # GEFS initialization folders look like YYYYMMDDHH
    if (
        len(name) == 10
        and name.isdigit()
    ):
        dates.append(name)


print(f"\nYear: {YEAR}")
print(f"Available initializations: {len(dates)}")

print("\nFirst 30 dates:")

# ---------------------------------------------------------
# Select representative dates
# ---------------------------------------------------------

if not dates:
    print("No dates found.")
    raise SystemExit(1)


selected_dates = []

step = max(1, len(dates) // 20)

for i in range(0, len(dates), step):
    selected_dates.append(dates[i])

selected_dates = selected_dates[:20]


print("\n" + "=" * 70)
print("SELECTED MVP DATES")
print("=" * 70)

for date in selected_dates:
    print(date)

print(f"\nSelected: {len(selected_dates)}")