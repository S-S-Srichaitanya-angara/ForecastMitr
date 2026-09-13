import pandas as pd

from forecastmitr_engine import predict


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "data/final_test_predictions.csv"
FEATURE_FILE = "data/features_chennai.csv"
OUTPUT_FILE = "data/forecastmitr_test_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading final unseen test set...")

test_df = pd.read_csv(TEST_FILE)

print(f"Test cases: {len(test_df)}")


print("\nLoading complete feature dataset...")

features_df = pd.read_csv(FEATURE_FILE)

print(f"Feature cases: {len(features_df)}")


# ============================================================
# CONVERT TIMESTAMPS
# ============================================================

test_df["forecast_issue_time"] = pd.to_datetime(
    test_df["forecast_issue_time"]
)

test_df["target_time"] = pd.to_datetime(
    test_df["target_time"]
)

features_df["forecast_issue_time"] = pd.to_datetime(
    features_df["forecast_issue_time"]
)

features_df["target_time"] = pd.to_datetime(
    features_df["target_time"]
)


# ============================================================
# MODEL FEATURES REQUIRED BY FORECASTMITR
# ============================================================

MODEL_FEATURES = [
    "control_forecast",
    "ensemble_mean",
    "ensemble_median",
    "ensemble_min",
    "ensemble_max",
    "ensemble_spread",
    "forecast_time_hours",
    "month",
]


# ============================================================
# CHECK FEATURE DATASET
# ============================================================

required_columns = [
    "forecast_issue_time",
    "target_time",
] + MODEL_FEATURES


missing_columns = [
    column
    for column in required_columns
    if column not in features_df.columns
]

if missing_columns:
    raise ValueError(
        "The feature dataset is missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# EXTRACT ONLY COMPLETE FEATURES
# ============================================================

feature_subset = features_df[
    [
        "forecast_issue_time",
        "target_time",
    ] + MODEL_FEATURES
].copy()


# ============================================================
# CHECK FEATURE UNIQUENESS
# ============================================================

duplicate_features = feature_subset.duplicated(
    subset=[
        "forecast_issue_time",
        "target_time",
    ]
).sum()

if duplicate_features > 0:
    raise ValueError(
        f"Found {duplicate_features} duplicate "
        "forecast/target pairs in features_chennai.csv."
    )


# ============================================================
# MERGE
# ============================================================

print("\nMatching test cases with complete model features...")

# IMPORTANT:
# Only merge the timestamp keys from the test dataset.
# All model features come from features_chennai.csv.
#
# This prevents pandas from creating _test / _features
# column suffixes.

test_keys = test_df[
    [
        "forecast_issue_time",
        "target_time",
    ]
].copy()


merged_df = test_keys.merge(
    feature_subset,
    on=[
        "forecast_issue_time",
        "target_time",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# VALIDATE MATCHING
# ============================================================

if len(merged_df) != len(test_df):
    raise ValueError(
        "The feature merge changed the number of test cases.\n"
        f"Expected: {len(test_df)}\n"
        f"Got:      {len(merged_df)}"
    )


missing_feature_rows = merged_df[
    MODEL_FEATURES
].isna().any(axis=1)


missing_count = int(
    missing_feature_rows.sum()
)


if missing_count > 0:

    print(
        f"\nERROR: {missing_count} test cases "
        "are missing model features."
    )

    print(
        merged_df.loc[
            missing_feature_rows,
            [
                "forecast_issue_time",
                "target_time",
            ],
        ].to_string(index=False)
    )

    raise ValueError(
        "Some test cases could not be matched "
        "to features_chennai.csv."
    )


print(
    f"Successfully matched "
    f"{len(merged_df)} / {len(test_df)} cases."
)


# ============================================================
# VERIFY FEATURE VALUES AGAINST ORIGINAL TEST FILE
# ============================================================

print("\nChecking forecast values...")

comparison = merged_df.merge(
    test_df[
        [
            "forecast_issue_time",
            "target_time",
            "control_forecast",
            "ensemble_mean",
            "ensemble_spread",
        ]
    ],
    on=[
        "forecast_issue_time",
        "target_time",
    ],
    how="inner",
    suffixes=(
        "_features",
        "_test",
    ),
    validate="one_to_one",
)


control_match = (
    comparison["control_forecast_features"]
    .round(8)
    ==
    comparison["control_forecast_test"]
    .round(8)
).all()


mean_match = (
    comparison["ensemble_mean_features"]
    .round(8)
    ==
    comparison["ensemble_mean_test"]
    .round(8)
).all()


spread_match = (
    comparison["ensemble_spread_features"]
    .round(8)
    ==
    comparison["ensemble_spread_test"]
    .round(8)
).all()


print(
    f"Control forecast match : "
    f"{control_match}"
)

print(
    f"Ensemble mean match    : "
    f"{mean_match}"
)

print(
    f"Ensemble spread match  : "
    f"{spread_match}"
)


if not (
    control_match
    and mean_match
    and spread_match
):
    raise ValueError(
        "Forecast values do not match between "
        "the two datasets."
    )


# ============================================================
# RUN FORECASTMITR
# ============================================================

print("\nRunning ForecastMitr inference engine...")

results = []


for _, row in merged_df.iterrows():

    result = predict(

        control_forecast=float(
            row["control_forecast"]
        ),

        ensemble_mean=float(
            row["ensemble_mean"]
        ),

        ensemble_median=float(
            row["ensemble_median"]
        ),

        ensemble_min=float(
            row["ensemble_min"]
        ),

        ensemble_max=float(
            row["ensemble_max"]
        ),

        forecast_time_hours=float(
            row["forecast_time_hours"]
        ),

        month=int(
            row["month"]
        ),
    )


    results.append({

        "forecast_issue_time":
            row["forecast_issue_time"],

        "target_time":
            row["target_time"],

        "bust_probability":
            result["bust_probability"],

        "predicted_bust":
            result["predicted_bust"],

        "risk_level":
            result["risk_level"],

        "diagnosis":
            result["diagnosis"],

        "primary_signal":
            result["primary_signal"],
    })


results_df = pd.DataFrame(results)


# ============================================================
# BASIC RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FORECASTMITR TEST RESULTS")
print("=" * 70)


predicted_busts = int(
    results_df["predicted_bust"].sum()
)


predicted_normal = int(
    (results_df["predicted_bust"] == 0).sum()
)


print(
    f"Total cases      : "
    f"{len(results_df)}"
)

print(
    f"Predicted busts  : "
    f"{predicted_busts}"
)

print(
    f"Predicted normal : "
    f"{predicted_normal}"
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RISK LEVEL DISTRIBUTION")
print("=" * 70)


risk_order = [
    "CRITICAL",
    "HIGH",
    "MODERATE",
    "LOW",
]


risk_counts = (
    results_df["risk_level"]
    .value_counts()
    .reindex(
        risk_order,
        fill_value=0,
    )
)


print(risk_counts)


# ============================================================
# DIAGNOSTIC DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC DISTRIBUTION")
print("=" * 70)


print(
    results_df["diagnosis"]
    .value_counts()
)


# ============================================================
# CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 70)
print("ENGINE CONSISTENCY CHECK")
print("=" * 70)


original_predictions = (
    test_df[
        [
            "forecast_issue_time",
            "target_time",
            "predicted_bust",
        ]
    ]
    .copy()
)


original_predictions["forecast_issue_time"] = pd.to_datetime(
    original_predictions["forecast_issue_time"]
)

original_predictions["target_time"] = pd.to_datetime(
    original_predictions["target_time"]
)


comparison = results_df.merge(
    original_predictions,
    on=[
        "forecast_issue_time",
        "target_time",
    ],
    how="inner",
    suffixes=(
        "_engine",
        "_original",
    ),
    validate="one_to_one",
)


prediction_matches = (
    comparison["predicted_bust_engine"]
    ==
    comparison["predicted_bust_original"]
)


matches = int(
    prediction_matches.sum()
)


total = len(comparison)


print(
    f"Matching predictions : "
    f"{matches} / {total}"
)


print(
    f"Consistency           : "
    f"{matches / total:.1%}"
)


if matches == total:

    print(
        "\nSUCCESS:"
        "\nForecastMitr reproduces "
        "the original model predictions "
        "for every test case."
    )

else:

    mismatches = comparison[
        ~prediction_matches
    ]

    print(
        "\nWARNING:"
        f"\n{len(mismatches)} predictions "
        "do not match."
    )

    print(
        "\nMismatched cases:"
    )

    print(
        mismatches[
            [
                "forecast_issue_time",
                "target_time",
                "predicted_bust_engine",
                "predicted_bust_original",
            ]
        ].to_string(index=False)
    )


# ============================================================
# TOP 10 RISK CASES
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 RISK CASES")
print("=" * 70)


top_cases = (
    results_df
    .sort_values(
        "bust_probability",
        ascending=False,
    )
    .head(10)
)


print(
    top_cases.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)
print(
    f"Saved to: {OUTPUT_FILE}"
)
print("=" * 70)