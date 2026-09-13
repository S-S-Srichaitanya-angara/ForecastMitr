import json
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = Path("models/forecastmitr_xgboost.joblib")
CONFIG_FILE = Path("models/forecastmitr_config.json")


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}\n"
        "Run build_forecastmitr_model.py first."
    )

if not CONFIG_FILE.exists():
    raise FileNotFoundError(
        f"Configuration not found: {CONFIG_FILE}\n"
        "Run build_forecastmitr_model.py first."
    )


model = joblib.load(MODEL_FILE)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)


FEATURES = config["features"]
DECISION_THRESHOLD = config["decision_threshold"]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    control_forecast,
    ensemble_mean,
    ensemble_median,
    ensemble_min,
    ensemble_max,
    forecast_time_hours,
    month,
):
    """
    Convert raw forecast information into the features
    expected by the trained XGBoost model.
    """

    ensemble_spread = ensemble_max - ensemble_min

    control_vs_ensemble = (
        control_forecast - ensemble_mean
    )

    relative_spread = (
        ensemble_spread / (ensemble_mean + 0.1)
    )

    return {
        "control_forecast": control_forecast,
        "ensemble_mean": ensemble_mean,
        "ensemble_median": ensemble_median,
        "ensemble_min": ensemble_min,
        "ensemble_max": ensemble_max,
        "ensemble_spread": ensemble_spread,
        "forecast_time_hours": forecast_time_hours,
        "month": month,
        "control_vs_ensemble": control_vs_ensemble,
        "relative_spread": relative_spread,
    }


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(bust_probability):
    """
    Prototype risk bands.

    IMPORTANT:
    These are decision bands, not calibrated probabilities.
    """

    if bust_probability >= 0.85:
        return "CRITICAL"

    if bust_probability >= 0.70:
        return "HIGH"

    if bust_probability >= 0.40:
        return "MODERATE"

    return "LOW"


# ============================================================
# DIAGNOSTIC ENGINE
# ============================================================

def diagnose_bust(
    control_forecast,
    ensemble_mean,
    ensemble_spread,
    observed_rainfall=None,
):
    """
    Explain the most likely reason for a forecast bust.

    Returns a dictionary because the main inference function
    expects the diagnosis to contain:

        type
        signal
        signals
        recommendation

    observed_rainfall is optional.

    For historical evaluation:
        observed_rainfall can be supplied.

    For real-time prediction:
        observed_rainfall should normally be None.
    """

    signals = []

    # ---------------------------------------------------------
    # 1. Historical observation-based diagnosis
    # ---------------------------------------------------------

    if observed_rainfall is not None:

        absolute_error = abs(
            control_forecast - observed_rainfall
        )

        # -----------------------------------------------------
        # Missed rainfall event
        # -----------------------------------------------------

        if (
            control_forecast < 1.0
            and observed_rainfall >= 1.0
        ):

            signals.append(
                "Observed rainfall substantially exceeds control forecast"
            )

            if ensemble_mean >= 1.0:
                signals.append(
                    "Ensemble consensus indicates a rainfall event"
                )

            return {
                "type": "POSSIBLE MISSED EVENT",

                "signal":
                    "Observed rainfall substantially exceeds control forecast",

                "signals": signals,

                "recommendation":
                    "Review the forecast because rainfall occurred "
                    "despite a weak deterministic forecast."
            }

        # -----------------------------------------------------
        # False alarm
        # -----------------------------------------------------

        if (
            control_forecast >= 1.0
            and observed_rainfall < 1.0
        ):

            signals.append(
                "Control forecast indicates rainfall but observation is weak"
            )

            if (
                ensemble_mean >= 0
                and control_forecast > 2 * max(ensemble_mean, 0.1)
            ):
                signals.append(
                    "Control forecast substantially exceeds ensemble consensus"
                )

            return {
                "type": "POSSIBLE FALSE ALARM",

                "signal":
                    "Control forecast indicates rainfall but observation is weak",

                "signals": signals,

                "recommendation":
                    "Treat the deterministic rainfall forecast with caution "
                    "and compare it with ensemble consensus."
            }

        # -----------------------------------------------------
        # Magnitude error
        # -----------------------------------------------------

        if (
            control_forecast >= 1.0
            and observed_rainfall >= 1.0
            and absolute_error >= 1.5
        ):

            signals.append(
                "Forecast and observation both indicate rainfall, "
                "but magnitude differs substantially"
            )

            return {
                "type": "MAGNITUDE ERROR",

                "signal":
                    "Large difference between forecast and observed rainfall",

                "signals": signals,

                "recommendation":
                    "Review the forecast magnitude against observations "
                    "and ensemble guidance."
            }

    # ---------------------------------------------------------
    # 2. Control vs ensemble disagreement
    # ---------------------------------------------------------

    if (
        control_forecast >= 1.0
        and ensemble_mean >= 0.1
        and control_forecast > 2 * ensemble_mean
    ):

        signals.append(
            "Control forecast substantially exceeds ensemble consensus"
        )

        if ensemble_spread >= 2.0:
            signals.append(
                "Moderate ensemble disagreement"
            )

        if ensemble_spread >= 5.0:
            signals.append(
                "Large ensemble disagreement"
            )

        return {
            "type": "POSSIBLE FALSE ALARM",

            "signal":
                "Control forecast substantially exceeds ensemble consensus",

            "signals": signals,

            "recommendation":
                "Treat the deterministic forecast with caution "
                "and consider ensemble uncertainty before issuing an alert."
        }

    # ---------------------------------------------------------
    # 3. Ensemble consensus substantially exceeds control
    # ---------------------------------------------------------

    if (
        ensemble_mean >= 1.0
        and ensemble_mean > 2 * max(control_forecast, 0.1)
    ):

        signals.append(
            "Ensemble consensus substantially exceeds control forecast"
        )

        return {
            "type": "POSSIBLE MISSED EVENT",

            "signal":
                "Ensemble consensus substantially exceeds control forecast",

            "signals": signals,

            "recommendation":
                "Consider the possibility of an underestimated rainfall "
                "event and review the ensemble guidance."
        }

    # ---------------------------------------------------------
    # 4. Large ensemble disagreement
    # ---------------------------------------------------------

    if ensemble_spread >= 5.0:

        signals.append(
            "Large ensemble disagreement"
        )

        return {
            "type": "HIGH ENSEMBLE UNCERTAINTY",

            "signal":
                "Large ensemble disagreement",

            "signals": signals,

            "recommendation":
                "Forecast confidence is low. Review the ensemble members "
                "before making a high-impact decision."
        }

    # ---------------------------------------------------------
    # 5. High forecast intensity
    # ---------------------------------------------------------

    if control_forecast >= 3.0:

        signals.append(
            "High control forecast intensity"
        )

        return {
            "type": "HIGH FORECAST INTENSITY",

            "signal":
                "High control forecast intensity",

            "signals": signals,

            "recommendation":
                "Monitor this forecast closely and compare it with "
                "ensemble guidance."
        }

    # ---------------------------------------------------------
    # 6. Moderate ensemble disagreement
    # ---------------------------------------------------------

    if ensemble_spread >= 2.0:

        signals.append(
            "Moderate ensemble disagreement"
        )

        return {
            "type": "MODERATE ENSEMBLE UNCERTAINTY",

            "signal":
                "Moderate ensemble disagreement",

            "signals": signals,

            "recommendation":
                "Consider ensemble spread when assessing forecast confidence."
        }

    # ---------------------------------------------------------
    # 7. Weak signal
    # ---------------------------------------------------------

    signals.append(
        "No dominant physical disagreement identified"
    )

    return {
        "type": "WEAK BUST SIGNAL",

        "signal":
            "Weak bust signal",

        "signals": signals,

        "recommendation":
            "No dominant forecast disagreement was identified. "
            "Continue normal monitoring."
    }


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def predict(
    control_forecast,
    ensemble_mean,
    ensemble_median,
    ensemble_min,
    ensemble_max,
    forecast_time_hours,
    month,
    observed_rainfall=None,
):
    """
    Run the complete ForecastMitr inference pipeline.

    Parameters
    ----------
    control_forecast : float
        GEFS control precipitation forecast in mm/3h.

    ensemble_mean : float
        Mean precipitation across ensemble members.

    ensemble_median : float
        Median precipitation across ensemble members.

    ensemble_min : float
        Minimum ensemble precipitation.

    ensemble_max : float
        Maximum ensemble precipitation.

    forecast_time_hours : float
        Forecast lead time.

    month : int
        Month number, 1-12.

    observed_rainfall : float, optional
        Actual rainfall. Used only for historical evaluation.
        Should NOT be supplied for genuine real-time prediction.
    """

    # --------------------------------------------------------
    # Create model features
    # --------------------------------------------------------

    feature_dict = create_features(
        control_forecast=control_forecast,
        ensemble_mean=ensemble_mean,
        ensemble_median=ensemble_median,
        ensemble_min=ensemble_min,
        ensemble_max=ensemble_max,
        forecast_time_hours=forecast_time_hours,
        month=month,
    )

    X = pd.DataFrame(
        [feature_dict],
        columns=FEATURES,
    )

    # --------------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------------

    bust_probability = float(
        model.predict_proba(X)[0, 1]
    )

    predicted_bust = int(
        bust_probability >= DECISION_THRESHOLD
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    risk_level = get_risk_level(
        bust_probability
    )

    # --------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------

    diagnosis = diagnose_bust(
        control_forecast=control_forecast,
        ensemble_mean=ensemble_mean,
        ensemble_spread=feature_dict["ensemble_spread"],
        observed_rainfall=observed_rainfall,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {
        "bust_probability": round(
            bust_probability, 4
        ),

        "predicted_bust": predicted_bust,

        "risk_level": risk_level,

        "diagnosis": diagnosis["type"],

        "primary_signal": diagnosis["signal"],

        "signals": diagnosis["signals"],

        "recommendation": diagnosis["recommendation"],

        "forecast": {
            "control_mm_3h": round(
                control_forecast, 3
            ),

            "ensemble_mean_mm_3h": round(
                ensemble_mean, 3
            ),

            "ensemble_median_mm_3h": round(
                ensemble_median, 3
            ),

            "ensemble_min_mm_3h": round(
                ensemble_min, 3
            ),

            "ensemble_max_mm_3h": round(
                ensemble_max, 3
            ),

            "ensemble_spread_mm": round(
                feature_dict["ensemble_spread"],
                3,
            ),

            "lead_time_hours": forecast_time_hours,

            "month": month,
        },
    }

    return result


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FORECASTMITR INFERENCE ENGINE")
    print("=" * 70)

    result = predict(
        control_forecast=5.30,
        ensemble_mean=1.62,
        ensemble_median=0.72,
        ensemble_min=0.0,
        ensemble_max=5.30,
        forecast_time_hours=72,
        month=11,
    )

    print("\nRESULT")
    print("=" * 70)

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

    print("\nForecast:")

    print(
        f"  Control        : "
        f"{result['forecast']['control_mm_3h']:.2f} mm/3h"
    )

    print(
        f"  Ensemble mean  : "
        f"{result['forecast']['ensemble_mean_mm_3h']:.2f} mm/3h"
    )

    print(
        f"  Ensemble spread: "
        f"{result['forecast']['ensemble_spread_mm']:.2f} mm"
    )

    print("=" * 70)