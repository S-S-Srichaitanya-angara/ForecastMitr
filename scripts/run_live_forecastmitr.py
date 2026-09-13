from datetime import datetime

from forecastmitr_engine import predict


# ============================================================
# LIVE GEFS VALUES
# ============================================================

MEMBER_VALUES = {
    "c00": 0.03,
    "p01": 0.00,
    "p02": 0.18,
    "p03": 0.22,
    "p04": 0.04,
}


# ============================================================
# FORECAST METADATA
# ============================================================

FORECAST_ISSUE_TIME = "2026-09-11 00:00"
TARGET_TIME = "2026-09-11 03:00"

LEAD_HOURS = 3
MONTH = 9


# ============================================================
# ENSEMBLE STATISTICS
# ============================================================

values = list(MEMBER_VALUES.values())

control_forecast = MEMBER_VALUES["c00"]

ensemble_mean = sum(values) / len(values)

sorted_values = sorted(values)

if len(sorted_values) % 2 == 1:
    ensemble_median = sorted_values[len(sorted_values) // 2]
else:
    middle = len(sorted_values) // 2
    ensemble_median = (
        sorted_values[middle - 1]
        + sorted_values[middle]
    ) / 2

ensemble_min = min(values)
ensemble_max = max(values)

ensemble_spread = ensemble_max - ensemble_min


# ============================================================
# RUN FORECASTMITR
# ============================================================

result = predict(
    control_forecast=control_forecast,
    ensemble_mean=ensemble_mean,
    ensemble_median=ensemble_median,
    ensemble_min=ensemble_min,
    ensemble_max=ensemble_max,
    forecast_time_hours=LEAD_HOURS,
    month=MONTH,
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("=" * 70)
print("FORECASTMITR LIVE INFERENCE")
print("=" * 70)

print(f"\nIssue time : {FORECAST_ISSUE_TIME}")
print(f"Target time: {TARGET_TIME}")
print(f"Lead time  : {LEAD_HOURS} hours")

print("\nLIVE GEFS ENSEMBLE")
print("-" * 70)

for member, rainfall in MEMBER_VALUES.items():
    print(f"{member:>4} : {rainfall:.3f} mm/3h")

print("\nENSEMBLE STATISTICS")
print("-" * 70)

print(f"Mean   : {ensemble_mean:.3f} mm/3h")
print(f"Median : {ensemble_median:.3f} mm/3h")
print(f"Min    : {ensemble_min:.3f} mm/3h")
print(f"Max    : {ensemble_max:.3f} mm/3h")
print(f"Spread : {ensemble_spread:.3f} mm")

print("\nFORECASTMITR RESULT")
print("-" * 70)

print(
    f"Bust score : "
    f"{result['bust_probability']:.1%}"
)

print(
    f"Risk level : "
    f"{result['risk_level']}"
)

print(
    f"Bust       : "
    f"{'YES' if result['predicted_bust'] else 'NO'}"
)

print(
    f"Diagnosis  : "
    f"{result['diagnosis']}"
)

print(
    f"\nPrimary signal:"
    f"\n{result['primary_signal']}"
)

print("\nSignals:")

for signal in result["signals"]:
    print(f"  - {signal}")

print(
    f"\nRecommendation:"
    f"\n{result['recommendation']}"
)

print("\n" + "=" * 70)
print("LIVE INFERENCE SUCCESS")
print("=" * 70)