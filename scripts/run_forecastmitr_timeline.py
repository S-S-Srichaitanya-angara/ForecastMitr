from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from forecastmitr_engine import predict


# ============================================================
# CONFIGURATION
# ============================================================

ENSEMBLE_FILE = Path("data/ensemble_chennai.csv")

MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


# ============================================================
# LOAD DATA
# ============================================================

if not ENSEMBLE_FILE.exists():
    raise FileNotFoundError(
        f"File not found: {ENSEMBLE_FILE}"
    )

df = pd.read_csv(ENSEMBLE_FILE)

df["forecast_issue_time"] = pd.to_datetime(
    df["forecast_issue_time"]
)

df["target_time"] = pd.to_datetime(
    df["target_time"]
)


# ============================================================
# SELECT ONE GEFS INITIALIZATION
# ============================================================

issue_time = df["forecast_issue_time"].min()

df_case = df[
    df["forecast_issue_time"] == issue_time
].copy()

if df_case.empty:
    raise ValueError(
        "No data found for selected initialization."
    )


# ============================================================
# VALIDATE ENSEMBLE MEMBERS
# ============================================================

available_members = set(
    df_case["ensemble_member"]
)

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
# FORECAST LEADS
# ============================================================

leads = sorted(
    df_case["forecast_time_hours"]
    .unique()
)

print()
print("=" * 80)
print("FORECASTMITR MULTI-LEAD INFERENCE")
print("=" * 80)

print(f"Initialization : {issue_time}")
print("Location       : Chennai")
print(f"Forecast leads : {len(leads)}")


# ============================================================
# RUN FORECASTMITR FOR EVERY LEAD
# ============================================================

results = []

for lead_time in leads:

    lead_case = df_case[
        df_case["forecast_time_hours"]
        == lead_time
    ].copy()

    # --------------------------------------------------------
    # Make sure all five members exist
    # --------------------------------------------------------

    available = set(
        lead_case["ensemble_member"]
    )

    if not all(
        member in available
        for member in MEMBERS
    ):
        print(
            f"Skipping {lead_time}h: "
            "missing ensemble member"
        )
        continue

    # --------------------------------------------------------
    # Extract member forecasts
    # --------------------------------------------------------

    member_values = {}

    for member in MEMBERS:

        row = lead_case[
            lead_case["ensemble_member"]
            == member
        ]

        if len(row) != 1:
            raise ValueError(
                f"Expected one row for {member} "
                f"at lead {lead_time}h, "
                f"found {len(row)}"
            )

        member_values[member] = float(
            row["forecast_rainfall_mm"].iloc[0]
        )

    rainfall = list(
        member_values.values()
    )

    # --------------------------------------------------------
    # Ensemble statistics
    # --------------------------------------------------------

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

    end_step = float(
        lead_case["end_step_hours"].iloc[0]
    )

    target_time = (
        issue_time
        + pd.Timedelta(
            hours=end_step
        )
    )

    month = int(
        target_time.month
    )

    # --------------------------------------------------------
    # ForecastMitr inference
    # --------------------------------------------------------

    result = predict(
        control_forecast=member_values["c00"],
        ensemble_mean=ensemble_mean,
        ensemble_median=ensemble_median,
        ensemble_min=ensemble_min,
        ensemble_max=ensemble_max,
        forecast_time_hours=float(
            lead_time
        ),
        month=month,
    )

    results.append(
        {
            "forecast_issue_time": issue_time,
            "target_time": target_time,
            "forecast_time_hours": lead_time,

            "control_forecast": (
                member_values["c00"]
            ),

            "ensemble_mean": ensemble_mean,
            "ensemble_median": ensemble_median,
            "ensemble_min": ensemble_min,
            "ensemble_max": ensemble_max,
            "ensemble_spread": ensemble_spread,

            "bust_probability": (
                result["bust_probability"]
            ),

            "predicted_bust": (
                result["predicted_bust"]
            ),

            "risk_level": (
                result["risk_level"]
            ),

            "diagnosis": (
                result["diagnosis"]
            ),

            "primary_signal": (
                result["primary_signal"]
            ),

            "recommendation": (
                result["recommendation"]
            ),
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError(
        "No ForecastMitr predictions were generated."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = Path(
    "data/forecastmitr_timeline.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY TIMELINE
# ============================================================

print()
print("=" * 80)
print("FORECAST RISK TIMELINE")
print("=" * 80)

print(
    f"{'Lead':>6}  "
    f"{'Target':<19}  "
    f"{'Bust':>8}  "
    f"{'Risk':<9}  "
    f"Diagnosis"
)

print("-" * 80)

for _, row in results_df.iterrows():

    print(
        f"{int(row['forecast_time_hours']):>4}h  "
        f"{str(row['target_time']):<19}  "
        f"{row['bust_probability']:>7.1%}  "
        f"{row['risk_level']:<9}  "
        f"{row['diagnosis']}"
    )


# ============================================================
# HIGHEST-RISK WINDOWS
# ============================================================

top_results = results_df.sort_values(
    "bust_probability",
    ascending=False
).head(5)


print()
print("=" * 80)
print("TOP 5 HIGHEST-RISK FORECAST WINDOWS")
print("=" * 80)

for _, row in top_results.iterrows():

    print()
    print(
        f"Lead       : "
        f"{int(row['forecast_time_hours'])} h"
    )

    print(
        f"Target     : "
        f"{row['target_time']}"
    )

    print(
        f"Bust score : "
        f"{row['bust_probability']:.1%}"
    )

    print(
        f"Risk       : "
        f"{row['risk_level']}"
    )

    print(
        f"Diagnosis  : "
        f"{row['diagnosis']}"
    )

    print(
        f"Signal     : "
        f"{row['primary_signal']}"
    )


print()
print("=" * 80)

print(
    f"Total forecast windows : "
    f"{len(results_df)}"
)

print(
    f"Predicted bust windows : "
    f"{results_df['predicted_bust'].sum()}"
)

print(
    f"Saved to               : "
    f"{output_file}"
)

print("=" * 80)