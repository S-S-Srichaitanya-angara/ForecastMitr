import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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

DATA_FILE = "data/features_chennai.csv"


def create_labels(df, threshold):
    df = df.copy()

    df["absolute_control_error"] = (
        df["control_forecast"]
        - df["observed_rainfall_mm"]
    ).abs()

    event_significant = (
        (df["observed_rainfall_mm"] >= 1.0)
        | (df["control_forecast"] >= 1.0)
    )

    df["bust"] = (
        (df["absolute_control_error"] >= threshold)
        & event_significant
    ).astype(int)

    return df


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    df["forecast_issue_time"] = pd.to_datetime(
        df["forecast_issue_time"]
    )

    init_dates = sorted(
        df["forecast_issue_time"].unique()
    )

    print(f"Cases: {len(df)}")
    print(f"Initialization dates: {len(init_dates)}")

    # ---------------------------------------------------------
    # Use the same rolling-origin setup as our robustness test
    # ---------------------------------------------------------

    min_train_dates = 10

    all_predictions = []

    for i in range(min_train_dates, len(init_dates)):

        train_dates = init_dates[:i]
        test_date = init_dates[i]

        train = df[
            df["forecast_issue_time"].isin(train_dates)
        ].copy()

        test = df[
            df["forecast_issue_time"] == test_date
        ].copy()

        # Calculate threshold ONLY from training data.
        train_errors = (
            train["control_forecast"]
            - train["observed_rainfall_mm"]
        ).abs()

        label_threshold = train_errors.quantile(0.90)

        train = create_labels(
            train,
            label_threshold,
        )

        test = create_labels(
            test,
            label_threshold,
        )

        X_train = train[FEATURES]
        y_train = train["bust"]

        X_test = test[FEATURES]
        y_test = test["bust"]

        if y_train.nunique() < 2:
            continue

        positive = y_train.sum()
        negative = len(y_train) - positive

        scale_pos_weight = (
            negative / positive
            if positive > 0
            else 1.0
        )

        model = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.2,
            reg_lambda=2,
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        fold_predictions = pd.DataFrame({
            "test_date": test_date,
            "actual": y_test.values,
            "probability": probabilities,
        })

        all_predictions.append(
            fold_predictions
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    print(
        f"\nTotal out-of-sample predictions: "
        f"{len(predictions)}"
    )

    # ---------------------------------------------------------
    # Test classification thresholds
    # ---------------------------------------------------------

    thresholds = np.round(
        np.arange(0.50, 1.00, 0.01),
        2
    )

    results = []

    for threshold in thresholds:

        predicted = (
            predictions["probability"]
            >= threshold
        ).astype(int)

        actual = predictions["actual"]

        precision = precision_score(
            actual,
            predicted,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0,
        )

        tp = (
            (predicted == 1)
            & (actual == 1)
        ).sum()

        fp = (
            (predicted == 1)
            & (actual == 0)
        ).sum()

        fn = (
            (predicted == 0)
            & (actual == 1)
        ).sum()

        tn = (
            (predicted == 0)
            & (actual == 0)
        ).sum()

        results.append({
            "decision_threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # Best F1
    best = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n" + "=" * 80)
    print("BEST F1 THRESHOLD")
    print("=" * 80)

    print(
        f"Threshold : {best['decision_threshold']:.2f}"
    )
    print(
        f"Precision : {best['precision']:.3f}"
    )
    print(
        f"Recall    : {best['recall']:.3f}"
    )
    print(
        f"F1        : {best['f1']:.3f}"
    )

    output_file = (
        "data/threshold_optimization.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved results to: {output_file}"
    )


if __name__ == "__main__":
    main()