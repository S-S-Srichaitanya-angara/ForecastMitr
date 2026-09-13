from pathlib import Path
import sys

import pandas as pd

# Allow importing forecastmitr_engine.py from scripts/
sys.path.append(str(Path(__file__).resolve().parent))

from forecastmitr_engine import predict


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data/ensemble_chennai.csv")

MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


# ============================================================
# LOAD GEFS DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"GEFS dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

df["forecast_issue_time"] = pd.to_datetime(
    df["forecast_issue_time"]
)

df["target_time"] = pd.to_datetime(
    df["target_time"]
)


# ============================================================
# SELECT ONE FORECAST CASE
# ============================================================

issue_time = df["forecast_issue_time"].min()

cases = df[
    df["forecast_issue_time"] == issue_time
].copy()

if cases.empty:
    raise ValueError(
        "No forecast cases found for selected initialization."
    )


target_time = cases["target_time"].iloc[0]

case = cases[
    cases["target_time"] == target_time
].copy()


# ============================================================
# VERIFY ENSEMBLE MEMBERS
# ============================================================

available_members = set(case["ensemble_member"])

missing_members = [
    member
    for member in MEMBERS
    if member not in available_members
]

if missing_members:
    raise ValueError(
        f"Missing ensemble members: {missing_members}"
    )


# ============================================================
# EXTRACT MEMBER FORECASTS
# ============================================================

member_values = {}

for member in MEMBERS:

    row = case[
        case["ensemble_member"] == member
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one row for {member}, "
            f"found {len(row)}"
        )

    member_values[member] = float(
        row["forecast_rainfall_mm"].iloc[0]
    )


# ============================================================
# CALCULATE ENSEMBLE STATISTICS
# ============================================================

values = list(member_values.values())

ensemble_mean = sum(values) / len(values)

ensemble_median = sorted(values)[
    len(values) // 2
]

ensemble_min = min(values)

ensemble_max = max(values)

ensemble_spread = (
    ensemble_max - ensemble_min
)


# ============================================================
# FORECAST LEAD TIME
# ============================================================

forecast_time_hours = float(
    case["forecast_time_hours"].iloc[0]
)

month = int(
    target_time.month
)


# ============================================================
# RUN FORECASTMITR
# ============================================================

result = predict(
    control_forecast=member_values["c00"],
    ensemble_mean=ensemble_mean,
    ensemble_median=ensemble_median,
    ensemble_min=ensemble_min,
    ensemble_max=ensemble_max,
    forecast_time_hours=forecast_time_hours,
    month=month,
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 70)
print("FORECASTMITR SINGLE-CASE INFERENCE")
print("=" * 70)

print(f"Location       : Chennai")
print(f"Issue time     : {issue_time}")
print(f"Target time    : {target_time}")
print(f"Lead time      : {forecast_time_hours:.0f} hours")

print()
print("GEFS ENSEMBLE")
print("-" * 70)

for member, value in member_values.items():
    print(f"{member:15}: {value:.3f} mm/3h")

print(f"{'Mean':15}: {ensemble_mean:.3f} mm/3h")
print(f"{'Median':15}: {ensemble_median:.3f} mm/3h")
print(f"{'Min':15}: {ensemble_min:.3f} mm/3h")
print(f"{'Max':15}: {ensemble_max:.3f} mm/3h")
print(f"{'Spread':15}: {ensemble_spread:.3f} mm")

print()
print("FORECASTMITR RESULT")
print("-" * 70)

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
print("Primary signal:")
print(f"  {result['primary_signal']}")

print()
print("Signals:")

for signal in result["signals"]:
    print(f"  - {signal}")

print()
print("Recommendation:")
print(f"  {result['recommendation']}")

print("=" * 70)