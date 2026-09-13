import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "features_chennai.csv"
)


df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "forecast_issue_time",
        "target_time",
    ],
)


FEATURES = [
    "control_forecast",
    "ensemble_mean",
    "ensemble_median",
    "ensemble_min",
    "ensemble_max",
    "ensemble_spread",
    "forecast_time_hours",
    "month",
    "control_vs_ensemble",
    "relative_spread",
]


print("=" * 70)
print("FEATURE SIGNAL ANALYSIS")
print("=" * 70)

print(f"Total cases: {len(df)}")
print(f"Bust cases: {(df['bust'] == 1).sum()}")
print(f"Normal cases: {(df['bust'] == 0).sum()}")


# ------------------------------------------------------------
# GROUP STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION: NORMAL vs BUST")
print("=" * 70)

summary = []

for feature in FEATURES:

    normal = df.loc[
        df["bust"] == 0,
        feature
    ]

    bust = df.loc[
        df["bust"] == 1,
        feature
    ]

    summary.append({
        "feature": feature,

        "normal_mean": normal.mean(),
        "bust_mean": bust.mean(),

        "normal_median": normal.median(),
        "bust_median": bust.median(),

        "normal_std": normal.std(),
        "bust_std": bust.std(),
    })


summary_df = pd.DataFrame(summary)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# CORRELATION WITH BUST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION WITH BUST")
print("=" * 70)

correlations = []

for feature in FEATURES:

    correlation = df[
        [feature, "bust"]
    ].corr().iloc[0, 1]

    correlations.append({
        "feature": feature,
        "correlation": correlation,
        "absolute_correlation": abs(correlation),
    })


correlation_df = (
    pd.DataFrame(correlations)
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
)


print(
    correlation_df[
        ["feature", "correlation"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# BUST RATE BY LEAD TIME
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BUST RATE BY LEAD TIME")
print("=" * 70)

lead_stats = (
    df.groupby("forecast_time_hours")
    .agg(
        cases=("bust", "size"),
        busts=("bust", "sum"),
    )
)

lead_stats["bust_rate_percent"] = (
    100
    * lead_stats["busts"]
    / lead_stats["cases"]
)

print(
    lead_stats.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# BUST RATE BY MONTH
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BUST RATE BY MONTH")
print("=" * 70)

month_stats = (
    df.groupby("month")
    .agg(
        cases=("bust", "size"),
        busts=("bust", "sum"),
    )
)

month_stats["bust_rate_percent"] = (
    100
    * month_stats["busts"]
    / month_stats["cases"]
)

print(
    month_stats.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# ENSEMBLE UNCERTAINTY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENSEMBLE SPREAD: NORMAL vs BUST")
print("=" * 70)

for label, name in [(0, "NORMAL"), (1, "BUST")]:

    values = df.loc[
        df["bust"] == label,
        "ensemble_spread"
    ]

    print(f"\n{name}")

    print(
        f"Mean   : {values.mean():.4f}"
    )

    print(
        f"Median : {values.median():.4f}"
    )

    print(
        f"75%    : {values.quantile(0.75):.4f}"
    )

    print(
        f"90%    : {values.quantile(0.90):.4f}"
    )


# ------------------------------------------------------------
# EXTREME ENSEMBLE DISAGREEMENT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HIGH ENSEMBLE SPREAD CASES")
print("=" * 70)

high_spread = (
    df.sort_values(
        "ensemble_spread",
        ascending=False
    )
    .head(20)
)

columns = [
    "forecast_issue_time",
    "target_time",
    "forecast_time_hours",
    "control_forecast",
    "ensemble_mean",
    "ensemble_spread",
    "bust",
    "bust_type",
]

print(
    high_spread[columns]
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)