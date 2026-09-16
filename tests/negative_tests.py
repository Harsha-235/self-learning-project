"""
Mandatory Negative-Test and Recovery Campaign

NT-1: Excellent validation performance collapses under deployment-like conditions.
NT-2: Hidden leakage is exploited by the model.
NT-3: Temporal/group split violation is detected.
NT-4: Simulation performance does not transfer to representative conditions.
NT-5: Nominal optimization improves while worst-case risk increases.

These tests are controlled software/data fixtures for validating
the leakage-resistant ML workflow.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

RESULT_FILE = EXPERIMENTS_DIR / "mandatory_negative_test_results.csv"
LOG_FILE = EXPERIMENTS_DIR / "mandatory_negative_test_log.json"


RANDOM_STATE = 42


def make_base_dataset(n=2000):
    """Create a deterministic machine-like binary classification dataset."""

    rng = np.random.default_rng(RANDOM_STATE)

    temperature = rng.normal(60, 8, n)
    vibration = rng.normal(1.0, 0.25, n)
    pressure = rng.normal(100, 2, n)
    load = rng.normal(55, 12, n)

    risk = (
        0.05 * (temperature - 60)
        + 2.0 * (vibration - 1.0)
        + 0.02 * (load - 55)
    )

    probability = 1 / (1 + np.exp(-risk))
    failure = (rng.random(n) < probability).astype(int)

    return pd.DataFrame(
        {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "load_percent": load,
            "failure": failure,
        }
    )


def evaluate_model(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


# ============================================================
# NT-1
# Excellent validation performance collapses under deployment
# ============================================================

def test_nt1():
    rng = np.random.default_rng(RANDOM_STATE)

    n = 2000

    timestamp = pd.date_range(
        "2025-01-01",
        periods=n,
        freq="h"
    )

    # Feature whose relationship changes after the deployment boundary.
    signal = rng.normal(0, 1, n)

    failure = np.zeros(n, dtype=int)

    # Development period
    development = np.arange(n) < 1500

    failure[development] = (
        signal[development] > 0
    ).astype(int)

    # Deployment period has reversed relationship.
    failure[~development] = (
        signal[~development] < 0
    ).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "signal": signal,
            "failure": failure,
        }
    )

    # Random validation
    X = df[["signal"]]
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    random_result = evaluate_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Deployment-like temporal evaluation
    train = df.iloc[:1500]
    deployment = df.iloc[1500:]

    temporal_result = evaluate_model(
        train[["signal"]],
        deployment[["signal"]],
        train["failure"],
        deployment["failure"],
    )

    collapse_detected = (
        random_result["accuracy"] - temporal_result["accuracy"]
        >= 0.30
    )

    return {
        "test": "NT-1",
        "description": (
            "Validation performance appears excellent but collapses "
            "under deployment-like conditions."
        ),
        "validation_accuracy": random_result["accuracy"],
        "deployment_accuracy": temporal_result["accuracy"],
        "failure_detected": collapse_detected,
        "safe_response": "Hold deployment and investigate validation boundary.",
        "status": "PASS" if collapse_detected else "FAIL",
    }


# ============================================================
# NT-2
# Hidden leakage exploited by model
# ============================================================

def test_nt2():
    df = make_base_dataset()

    rng = np.random.default_rng(RANDOM_STATE)

    # Controlled hidden indirect leakage.
    df["hidden_leakage"] = (
        df["failure"]
        + rng.normal(0, 0.03, len(df))
    )

    X = df.drop(columns=["failure"])
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    result = evaluate_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    leakage_auc = roc_auc_score(
        y,
        df["hidden_leakage"]
    )

    detected = leakage_auc >= 0.95

    return {
        "test": "NT-2",
        "description": (
            "Hidden target-related information is introduced into "
            "a feature and must be exposed."
        ),
        "model_accuracy_with_leakage": result["accuracy"],
        "leakage_feature_auc": float(leakage_auc),
        "failure_detected": detected,
        "safe_response": "Reject the suspicious feature and retrain.",
        "status": "PASS" if detected else "FAIL",
    }


# ============================================================
# NT-3
# Invalid temporal/group split
# ============================================================

def test_nt3():
    timestamps = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="h"
    )

    groups = (
        ["M01"] * 25
        + ["M02"] * 25
        + ["M03"] * 25
        + ["M04"] * 25
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "machine_id": groups,
        }
    )

    # Deliberately invalid temporal split.
    training_end = df.iloc[60]["timestamp"]
    testing_start = df.iloc[55]["timestamp"]

    temporal_violation = testing_start <= training_end

    # Deliberately invalid group split.
    train_groups = set(df.iloc[:80]["machine_id"])
    test_groups = set(df.iloc[60:]["machine_id"])

    group_violation = bool(
        train_groups.intersection(test_groups)
    )

    detected = temporal_violation and group_violation

    return {
        "test": "NT-3",
        "description": (
            "Training and validation splits violate temporal or "
            "group structure."
        ),
        "temporal_violation_detected": temporal_violation,
        "group_violation_detected": group_violation,
        "failure_detected": detected,
        "safe_response": "Reject the invalid validation split.",
        "status": "PASS" if detected else "FAIL",
    }


# ============================================================
# NT-4
# Simulation does not transfer to representative conditions
# ============================================================

def test_nt4():
    """
    NT-4:
    Demonstrate that good simulation performance does not guarantee
    good performance under representative operating conditions.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    n_train = 1200
    n_test = 600

    # -----------------------------
    # Simulation environment
    # -----------------------------
    sim_temperature = rng.normal(50, 4, n_train)
    sim_vibration = rng.normal(0.8, 0.08, n_train)

    sim_failure = (
        (
            0.05 * (sim_temperature - 50)
            + 4 * (sim_vibration - 0.8)
            > 0
        )
    ).astype(int)

    simulation = pd.DataFrame(
        {
            "temperature": sim_temperature,
            "vibration": sim_vibration,
        }
    )

    # Train using simulation data.
    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(
        simulation,
        sim_failure
    )

    # Verify simulation performance.
    simulation_predictions = model.predict(simulation)

    simulation_accuracy = accuracy_score(
        sim_failure,
        simulation_predictions
    )

    # -----------------------------
    # Representative environment
    # -----------------------------
    representative_temperature = rng.normal(
        75,
        6,
        n_test
    )

    representative_vibration = rng.normal(
        1.5,
        0.15,
        n_test
    )

    representative = pd.DataFrame(
        {
            "temperature": representative_temperature,
            "vibration": representative_vibration,
        }
    )

    # Controlled domain shift:
    # the relationship between the sensors and failure
    # changes in the representative environment.
    representative_failure = (
        (
            0.05 * (representative_temperature - 75)
            + 4 * (representative_vibration - 1.5)
            < 0
        )
    ).astype(int)

    representative_predictions = model.predict(
        representative
    )

    representative_accuracy = accuracy_score(
        representative_failure,
        representative_predictions
    )

    # Simulation is good but representative transfer is poor.
    transfer_failure = (
        simulation_accuracy >= 0.90
        and representative_accuracy <= 0.60
    )

    return {
        "test": "NT-4",
        "description": (
            "Simulation performance does not transfer to "
            "representative operating conditions."
        ),
        "simulation_accuracy": float(
            simulation_accuracy
        ),
        "representative_accuracy": float(
            representative_accuracy
        ),
        "failure_detected": bool(
            transfer_failure
        ),
        "safe_response": (
            "Enter degraded/safe mode and require "
            "representative validation."
        ),
        "status": (
            "PASS"
            if transfer_failure
            else "FAIL"
        ),
    }
# ============================================================
# NT-5
# Optimization improves nominal performance but worsens risk
# ============================================================

def test_nt5():
    """
    NT-5:
    Demonstrate that an optimization can improve nominal performance
    while increasing worst-case risk.
    """

    # Controlled acceptance fixture.
    #
    # 0 = normal
    # 1 = failure

    failure = np.array([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
    ])

    # First 10 samples represent nominal operation.
    nominal = np.array([
        True, True, True, True, True,
        True, True, True, True, True,
        False, False, False, False, False,
        False, False, False, False, False,
    ])

    # Last 10 samples represent boundary / worst-case operation.
    worst_case = ~nominal

    # Baseline configuration:
    # conservative around the boundary,
    # but slightly less accurate in nominal operation.
    baseline_prediction = np.array([
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
    ])

    # Optimized configuration:
    # improves nominal performance,
    # but becomes unsafe on the worst-case boundary.
    optimized_prediction = np.array([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ])

    baseline_nominal_error = np.mean(
        baseline_prediction[nominal]
        != failure[nominal]
    )

    optimized_nominal_error = np.mean(
        optimized_prediction[nominal]
        != failure[nominal]
    )

    baseline_worst_error = np.mean(
        baseline_prediction[worst_case]
        != failure[worst_case]
    )

    optimized_worst_error = np.mean(
        optimized_prediction[worst_case]
        != failure[worst_case]
    )

    nominal_improved = (
        optimized_nominal_error
        < baseline_nominal_error
    )

    worst_case_worsened = (
        optimized_worst_error
        > baseline_worst_error
    )

    detected = (
        nominal_improved
        and worst_case_worsened
    )

    return {
        "test": "NT-5",
        "description": (
            "Optimization improves nominal performance "
            "while increasing worst-case risk."
        ),
        "baseline_nominal_error": float(
            baseline_nominal_error
        ),
        "optimized_nominal_error": float(
            optimized_nominal_error
        ),
        "baseline_worst_case_error": float(
            baseline_worst_error
        ),
        "optimized_worst_case_error": float(
            optimized_worst_error
        ),
        "nominal_improved": bool(
            nominal_improved
        ),
        "worst_case_worsened": bool(
            worst_case_worsened
        ),
        "failure_detected": bool(
            detected
        ),
        "safe_response": (
            "Reject the optimization and retain "
            "the safer configuration."
        ),
        "status": (
            "PASS"
            if detected
            else "FAIL"
        ),
    }

# ============================================================
# RUN ALL TESTS
# ============================================================

def main():
    results = [
        test_nt1(),
        test_nt2(),
        test_nt3(),
        test_nt4(),
        test_nt5(),
    ]

    df = pd.DataFrame(results)

    df.to_csv(
        RESULT_FILE,
        index=False,
    )

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            default=lambda obj: obj.item()
            if isinstance(obj, np.generic)
            else str(obj),
        )

    print("\n" + "=" * 70)
    print("MANDATORY NEGATIVE-TEST AND RECOVERY CAMPAIGN")
    print("=" * 70)

    for result in results:
        print(
            f"\n{result['test']} - {result['status']}"
        )
        print(
            f"Failure detected: "
            f"{result['failure_detected']}"
        )
        print(
            f"Safe response: "
            f"{result['safe_response']}"
        )

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    print("\n" + "=" * 70)
    print(
        f"RESULT: {passed}/5 mandatory negative tests PASS"
    )
    print("=" * 70)

    print(
        f"\nEvidence saved to:\n"
        f"{RESULT_FILE}\n"
        f"{LOG_FILE}"
    )

    return 0 if passed == 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())