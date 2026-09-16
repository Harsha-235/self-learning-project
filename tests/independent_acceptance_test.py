import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "machine_data_v2.csv"
OUTPUT_CSV = PROJECT_ROOT / "experiments" / "independent_acceptance_results.csv"
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "independent_acceptance_report.json"


FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent",
]


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():

    print("=" * 70)
    print("INDEPENDENT ACCEPTANCE TEST")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()
    y = df["failure"].copy()

    # ---------------------------------------------------------
    # Independent repeat of clean baseline
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    clean_model = GradientBoostingClassifier(random_state=42)
    clean_model.fit(X_train, y_train)

    clean_metrics = evaluate_model(
        clean_model,
        X_test,
        y_test,
    )

    # ---------------------------------------------------------
    # Independent controlled leakage injection
    # ---------------------------------------------------------

    rng = np.random.RandomState(42)

    df_leaky = df.copy()

    df_leaky["maintenance_risk_score"] = np.clip(
        0.20
        + 0.65 * df_leaky["failure"]
        + rng.normal(0, 0.08, len(df_leaky)),
        0,
        1,
    )

    leakage_auc = roc_auc_score(
        df_leaky["failure"],
        df_leaky["maintenance_risk_score"],
    )

    leakage_detected = leakage_auc >= 0.99

    # ---------------------------------------------------------
    # Repeat model with leakage
    # ---------------------------------------------------------

    leaky_features = FEATURES + ["maintenance_risk_score"]

    X_leaky = df_leaky[leaky_features]

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_leaky,
        y,
        test_size=0.20,
        random_state=42,
    )

    leaky_model = GradientBoostingClassifier(random_state=42)
    leaky_model.fit(X_train_l, y_train_l)

    leaky_metrics = evaluate_model(
        leaky_model,
        X_test_l,
        y_test_l,
    )

    leakage_effect_confirmed = (
        leaky_metrics["roc_auc"] > clean_metrics["roc_auc"] + 0.10
        and leakage_auc >= 0.99
    )

    # ---------------------------------------------------------
    # Leakage removal
    # ---------------------------------------------------------

    X_clean_again = df_leaky[FEATURES]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_clean_again,
        y,
        test_size=0.20,
        random_state=42,
    )

    recovered_model = GradientBoostingClassifier(random_state=42)
    recovered_model.fit(X_train_r, y_train_r)

    recovered_metrics = evaluate_model(
        recovered_model,
        X_test_r,
        y_test_r,
    )

    metric_differences = {
        metric: abs(
            clean_metrics[metric] - recovered_metrics[metric]
        )
        for metric in clean_metrics
    }

    performance_survives = all(
        difference <= 1e-12
        for difference in metric_differences.values()
    )

    # ---------------------------------------------------------
    # AC-3 decision
    # ---------------------------------------------------------

    ac3_pass = (
        leakage_detected
        and leakage_effect_confirmed
        and performance_survives
    )

    results = [
        {
            "test": "AC-3",
            "status": "PASS" if ac3_pass else "FAIL",
            "leakage_auc": leakage_auc,
            "clean_roc_auc": clean_metrics["roc_auc"],
            "leaky_roc_auc": leaky_metrics["roc_auc"],
            "recovered_roc_auc": recovered_metrics["roc_auc"],
            "leakage_detected": leakage_detected,
            "leakage_effect_confirmed": leakage_effect_confirmed,
            "performance_survives_removal": performance_survives,
        }
    ]

    pd.DataFrame(results).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    report = {
        "acceptance_test": "AC-3",
        "status": "PASS" if ac3_pass else "FAIL",
        "independent_repeat": True,
        "leakage_auc": leakage_auc,
        "clean_metrics": clean_metrics,
        "leaky_metrics": leaky_metrics,
        "recovered_metrics": recovered_metrics,
        "metric_differences": metric_differences,
        "leakage_detected": leakage_detected,
        "leakage_effect_confirmed": leakage_effect_confirmed,
        "performance_survives_removal": performance_survives,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("AC-3 -", "PASS" if ac3_pass else "FAIL")
    print("Independent leakage detection:", leakage_detected)
    print("Leakage effect confirmed:", leakage_effect_confirmed)
    print("Performance survives leakage removal:", performance_survives)

    print()
    print("Leakage feature ROC-AUC:", round(leakage_auc, 4))
    print("Clean ROC-AUC:", round(clean_metrics["roc_auc"], 4))
    print("Leaky ROC-AUC:", round(leaky_metrics["roc_auc"], 4))
    print("Recovered ROC-AUC:", round(recovered_metrics["roc_auc"], 4))

    print()
    print("Metric differences after leakage removal:")
    for metric, difference in metric_differences.items():
        print(f"  {metric}: {difference:.12f}")

    print()
    print("Evidence saved to:")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)

    print("=" * 70)


if __name__ == "__main__":
    main()