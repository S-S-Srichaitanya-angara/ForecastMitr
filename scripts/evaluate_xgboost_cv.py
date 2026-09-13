import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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
    """
    Create bust labels using ONLY the supplied threshold.

    Bust:
    - forecast error >= threshold
    - and the event is significant
    """

    df = df.copy()

    df["abs_control_error"] = (
        df["control_forecast"] - df["observed_rainfall_mm"]
    ).abs()

    event_significant = (
        (df["observed_rainfall_mm"] >= 1.0)
        | (df["control_forecast"] >= 1.0)
    )

    df["bust"] = (
        (df["abs_control_error"] >= threshold)
        & event_significant
    ).astype(int)

    return df


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    df["forecast_issue_time"] = pd.to_datetime(
        df["forecast_issue_time"]
    )

    df["target_time"] = pd.to_datetime(
        df["target_time"]
    )

    # Initialization date = independent forecast case.
    init_dates = sorted(
        df["forecast_issue_time"].unique()
    )

    print(f"Total cases: {len(df)}")
    print(f"Initialization dates: {len(init_dates)}")

    # ---------------------------------------------------------
    # Rolling-origin evaluation
    # ---------------------------------------------------------

    # We need enough dates for training.
    # Each iteration trains on earlier dates
    # and tests on the next unseen date.
    min_train_dates = 10

    results = []

    for i in range(min_train_dates, len(init_dates)):

        train_dates = init_dates[:i]
        test_date = init_dates[i]

        train = df[
            df["forecast_issue_time"].isin(train_dates)
        ].copy()

        test = df[
            df["forecast_issue_time"] == test_date
        ].copy()

        # -----------------------------------------------------
        # Leakage-free threshold
        # -----------------------------------------------------

        train_errors = (
            train["control_forecast"]
            - train["observed_rainfall_mm"]
        ).abs()

        threshold = train_errors.quantile(0.90)

        train = create_labels(train, threshold)
        test = create_labels(test, threshold)

        X_train = train[FEATURES]
        y_train = train["bust"]

        X_test = test[FEATURES]
        y_test = test["bust"]

        # Skip if training set has only one class.
        if y_train.nunique() < 2:
            print(
                f"Skipping {test_date}: "
                "training data has only one class."
            )
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

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        # ROC-AUC / PR-AUC require both classes.
        if y_test.nunique() == 2:
            roc_auc = roc_auc_score(
                y_test,
                probabilities,
            )

            pr_auc = average_precision_score(
                y_test,
                probabilities,
            )
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        results.append({
            "test_date": test_date,
            "train_dates": i,
            "threshold": threshold,
            "test_busts": int(y_test.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })

        roc_text = (
            f"{roc_auc:.3f}"
            if not np.isnan(roc_auc)
            else "N/A"
        )

        pr_text = (
            f"{pr_auc:.3f}"
            if not np.isnan(pr_auc)
            else "N/A"
        )

        print(
            f"{test_date.date()} | "
            f"threshold={threshold:.3f} | "
            f"busts={int(y_test.sum()):2d} | "
            f"P={precision:.3f} | "
            f"R={recall:.3f} | "
            f"F1={f1:.3f} | "
            f"ROC={roc_text} | "
            f"PR={pr_text}"
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_file = "data/xgboost_cv_results.csv"

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("ROLLING-ORIGIN EVALUATION")
    print("=" * 60)

    if len(results_df) == 0:
        print("No valid evaluation folds.")
        return

    metrics = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    print("\nMean ± standard deviation:")

    for metric in metrics:

        mean = results_df[metric].mean()
        std = results_df[metric].std()

        print(
            f"{metric.upper():10s}: "
            f"{mean:.3f} ± {std:.3f}"
        )

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()