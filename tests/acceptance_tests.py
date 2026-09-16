"""
Acceptance Test Harness

AC-1:
Representative operation - useful performance survives leakage removal.

AC-2:
Boundary/failure operation - leakage and invalid validation
conditions are detected safely.

AC-4:
Frozen resource envelope - reference and final comparisons
use the same approved environment, dataset, workload,
model configuration, and resource constraints.
"""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS_DIR = ROOT / "experiments"

CONTROLLED_LEAKAGE_FILE = (
    EXPERIMENTS_DIR
    / "controlled_indirect_leakage_results.csv"
)

NEGATIVE_TEST_FILE = (
    EXPERIMENTS_DIR
    / "mandatory_negative_test_results.csv"
)

ACCEPTANCE_RESULTS_FILE = (
    EXPERIMENTS_DIR
    / "acceptance_results.csv"
)

ACCEPTANCE_REPORT_FILE = (
    EXPERIMENTS_DIR
    / "acceptance_report.json"
)

RESOURCE_ENVELOPE_FILE = (
    EXPERIMENTS_DIR
    / "resource_envelope.json"
)


# ============================================================
# AC-1
# ============================================================

def run_ac1():
    """
    AC-1:
    Representative operation - useful performance survives
    leakage removal.
    """

    if not CONTROLLED_LEAKAGE_FILE.exists():
        raise FileNotFoundError(
            f"Required evidence file not found: "
            f"{CONTROLLED_LEAKAGE_FILE}"
        )

    results = pd.read_csv(CONTROLLED_LEAKAGE_FILE)

    # The controlled leakage experiment uses these columns.
    required_columns = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    missing_columns = (
        required_columns - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Required experiments.
    required_experiments = {
        "Clean Baseline",
        "Indirect Leakage",
        "After Leakage Removal",
    }

    available_experiments = set(results["model"])

    missing_experiments = (
        required_experiments - available_experiments
    )

    if missing_experiments:
        raise ValueError(
            "Missing required experiment results: "
            f"{sorted(missing_experiments)}"
        )

    clean = results[
        results["model"] == "Clean Baseline"
    ].iloc[0]

    leakage = results[
        results["model"] == "Indirect Leakage"
    ].iloc[0]

    recovered = results[
        results["model"] == "After Leakage Removal"
    ].iloc[0]

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    # Compare clean baseline with the model after
    # leakage removal.
    differences = {}

    for metric in metrics:
        differences[metric] = abs(
            float(clean[metric])
            - float(recovered[metric])
        )

    # Controlled experiment is deterministic.
    tolerance = 0.000001

    performance_survives = all(
        difference <= tolerance
        for difference in differences.values()
    )

    # Confirm that leakage had a measurable effect.
    leakage_had_effect = (
        float(leakage["roc_auc"])
        > float(clean["roc_auc"])
    )

    passed = (
        performance_survives
        and leakage_had_effect
    )

    return {
        "acceptance_test": "AC-1",

        "description": (
            "Representative operation: useful performance "
            "survives leakage removal."
        ),

        "clean_baseline": {
            metric: float(clean[metric])
            for metric in metrics
        },

        "indirect_leakage": {
            metric: float(leakage[metric])
            for metric in metrics
        },

        "after_leakage_removal": {
            metric: float(recovered[metric])
            for metric in metrics
        },

        "metric_differences": differences,

        "tolerance": tolerance,

        "leakage_effect_confirmed": leakage_had_effect,

        "performance_survives_leakage_removal": (
            performance_survives
        ),

        "status": "PASS" if passed else "FAIL",

        "safe_response": (
            "Accept representative performance evidence."
            if passed
            else
            "Reject acceptance evidence and investigate "
            "performance loss after leakage removal."
        ),
    }


# ============================================================
# AC-2
# ============================================================

def run_ac2():
    """
    AC-2:
    Boundary/failure operation.

    Verify that the mandatory negative-test campaign:
        1. Detects each failure condition.
        2. Produces PASS for each test.
        3. Provides a safe response.
        4. Specifically detects leakage.
        5. Detects invalid temporal/group validation.
        6. Detects lack of representative transfer.
        7. Detects increased worst-case risk.
    """

    if not NEGATIVE_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Required evidence file not found: "
            f"{NEGATIVE_TEST_FILE}"
        )

    results = pd.read_csv(NEGATIVE_TEST_FILE)

    # Required columns for AC-2.
    required_columns = {
        "test",
        "description",
        "failure_detected",
        "safe_response",
        "status",
    }

    missing_columns = (
        required_columns - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # All five mandatory negative tests must be present.
    required_tests = {
        "NT-1",
        "NT-2",
        "NT-3",
        "NT-4",
        "NT-5",
    }

    available_tests = set(results["test"])

    missing_tests = (
        required_tests - available_tests
    )

    if missing_tests:
        raise ValueError(
            "Missing mandatory negative tests: "
            f"{sorted(missing_tests)}"
        )

    # --------------------------------------------------------
    # Check that every mandatory test passed.
    # --------------------------------------------------------

    all_tests_pass = bool(
        results["status"]
        .astype(str)
        .str.upper()
        .eq("PASS")
        .all()
    )

    # --------------------------------------------------------
    # Check that every failure condition was actually detected.
    # --------------------------------------------------------

    all_failures_detected = bool(
        results["failure_detected"]
        .astype(bool)
        .all()
    )

    # --------------------------------------------------------
    # NT-2:
    # Hidden leakage must be detected.
    # --------------------------------------------------------

    nt2 = results[
        results["test"] == "NT-2"
    ].iloc[0]

    leakage_detected = bool(
        nt2["failure_detected"]
    )

    leakage_auc_available = (
        pd.notna(nt2["leakage_feature_auc"])
    )

    leakage_auc = (
        float(nt2["leakage_feature_auc"])
        if leakage_auc_available
        else None
    )

    # --------------------------------------------------------
    # NT-3:
    # Invalid temporal/group split must be detected.
    # --------------------------------------------------------

    nt3 = results[
        results["test"] == "NT-3"
    ].iloc[0]

    temporal_violation_detected = bool(
        nt3["temporal_violation_detected"]
    )

    group_violation_detected = bool(
        nt3["group_violation_detected"]
    )

    split_integrity_detected = (
        temporal_violation_detected
        and group_violation_detected
    )

    # --------------------------------------------------------
    # NT-4:
    # Simulation/test performance must not be accepted when
    # it fails to transfer to representative conditions.
    # --------------------------------------------------------

    nt4 = results[
        results["test"] == "NT-4"
    ].iloc[0]

    simulation_accuracy = float(
        nt4["simulation_accuracy"]
    )

    representative_accuracy = float(
        nt4["representative_accuracy"]
    )

    representative_transfer_failure_detected = (
        bool(nt4["failure_detected"])
        and
        representative_accuracy < simulation_accuracy
    )

    # --------------------------------------------------------
    # NT-5:
    # Optimization must be rejected when worst-case risk
    # increases even if nominal performance improves.
    # --------------------------------------------------------

    nt5 = results[
        results["test"] == "NT-5"
    ].iloc[0]

    nominal_improved = bool(
        nt5["nominal_improved"]
    )

    worst_case_worsened = bool(
        nt5["worst_case_worsened"]
    )

    unsafe_optimization_detected = (
        bool(nt5["failure_detected"])
        and
        nominal_improved
        and
        worst_case_worsened
    )

    # --------------------------------------------------------
    # AC-2 final decision.
    # --------------------------------------------------------

    passed = (
        all_tests_pass
        and
        all_failures_detected
        and
        leakage_detected
        and
        split_integrity_detected
        and
        representative_transfer_failure_detected
        and
        unsafe_optimization_detected
    )

    return {
        "acceptance_test": "AC-2",

        "description": (
            "Boundary/failure operation: leakage and "
            "invalid validation conditions are detected "
            "and handled safely."
        ),

        "mandatory_negative_tests": {
            "NT-1": {
                "status": str(nt1_status(results)),
                "failure_detected": bool(
                    get_test_row(
                        results,
                        "NT-1"
                    )["failure_detected"]
                ),
            },

            "NT-2": {
                "status": str(nt2["status"]),
                "failure_detected": leakage_detected,
                "leakage_feature_auc": leakage_auc,
            },

            "NT-3": {
                "status": str(nt3["status"]),
                "failure_detected": bool(
                    nt3["failure_detected"]
                ),
                "temporal_violation_detected": (
                    temporal_violation_detected
                ),
                "group_violation_detected": (
                    group_violation_detected
                ),
            },

            "NT-4": {
                "status": str(nt4["status"]),
                "failure_detected": bool(
                    nt4["failure_detected"]
                ),
                "simulation_accuracy": (
                    simulation_accuracy
                ),
                "representative_accuracy": (
                    representative_accuracy
                ),
            },

            "NT-5": {
                "status": str(nt5["status"]),
                "failure_detected": bool(
                    nt5["failure_detected"]
                ),
                "nominal_improved": nominal_improved,
                "worst_case_worsened": (
                    worst_case_worsened
                ),
            },
        },

        "all_tests_pass": all_tests_pass,

        "all_failures_detected": (
            all_failures_detected
        ),

        "leakage_detected": leakage_detected,

        "split_integrity_detected": (
            split_integrity_detected
        ),

        "representative_transfer_failure_detected": (
            representative_transfer_failure_detected
        ),

        "unsafe_optimization_detected": (
            unsafe_optimization_detected
        ),

        "status": "PASS" if passed else "FAIL",

        "safe_response": (
            "Accept boundary/failure evidence. "
            "Mandatory negative conditions were detected "
            "and safe responses were recorded."
            if passed
            else
            "Reject boundary/failure evidence and "
            "investigate undetected or incorrectly handled "
            "failure conditions."
        ),
    }


# ============================================================
# Helper functions for AC-2
# ============================================================

def get_test_row(results, test_id):
    """Return the row for a specific negative test."""

    matches = results[
        results["test"] == test_id
    ]

    if matches.empty:
        raise ValueError(
            f"Required test not found: {test_id}"
        )

    return matches.iloc[0]


def nt1_status(results):
    """Return NT-1 status."""

    return get_test_row(
        results,
        "NT-1"
    )["status"]


# ============================================================
# AC-4
# ============================================================

def run_ac4():
    """
    AC-4:
    Frozen resource envelope.

    Verify that the reference and final comparison use the same
    approved software environment, dataset, workload, model
    configuration, and resource constraints.
    """

    if not RESOURCE_ENVELOPE_FILE.exists():
        raise FileNotFoundError(
            f"Required resource envelope file not found: "
            f"{RESOURCE_ENVELOPE_FILE}"
        )

    with open(
        RESOURCE_ENVELOPE_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        envelope = json.load(file)

    # --------------------------------------------------------
    # Validate required sections.
    # --------------------------------------------------------

    required_sections = {
        "reference",
        "final",
        "resource_limits",
    }

    missing_sections = (
        required_sections - set(envelope.keys())
    )

    if missing_sections:
        raise ValueError(
            "Missing required resource envelope sections: "
            f"{sorted(missing_sections)}"
        )

    reference = envelope["reference"]
    final = envelope["final"]
    limits = envelope["resource_limits"]

    # --------------------------------------------------------
    # Check that the approved reference and final environments
    # are identical.
    # --------------------------------------------------------

    environment_fields = [
        "python_version",
        "pandas_version",
        "numpy_version",
        "scikit_learn_version",
        "joblib_version",
        "dataset",
        "dataset_rows",
        "feature_count",
        "target",
        "workload",
        "model",
    ]

    differences = {}

    for field in environment_fields:
        if reference.get(field) != final.get(field):
            differences[field] = {
                "reference": reference.get(field),
                "final": final.get(field),
            }

    environment_frozen = (
        len(differences) == 0
    )

    # --------------------------------------------------------
    # Check resource constraints.
    # --------------------------------------------------------

    resources_unchanged = (
        limits.get("hardware_change") is False
        and
        limits.get("workload_change") is False
        and
        limits.get("dataset_change") is False
        and
        limits.get("software_version_change") is False
    )

    passed = (
        environment_frozen
        and
        resources_unchanged
    )

    return {
        "acceptance_test": "AC-4",

        "description": (
            "Frozen resource envelope: reference and final "
            "comparisons use the same approved environment, "
            "dataset, workload, model configuration, and "
            "resource constraints."
        ),

        "environment_frozen": environment_frozen,

        "resources_unchanged": resources_unchanged,

        "differences": differences,

        "reference": reference,

        "final": final,

        "resource_limits": limits,

        "status": "PASS" if passed else "FAIL",

        "safe_response": (
            "Accept frozen resource-envelope evidence."
            if passed
            else
            "Reject the comparison and investigate changes "
            "to the approved resource envelope."
        ),
    }


# ============================================================
# Main Acceptance Harness
# ============================================================

def main():

    # --------------------------------------------------------
    # Run all acceptance tests.
    # --------------------------------------------------------

    ac1_result = run_ac1()

    ac2_result = run_ac2()

    ac4_result = run_ac4()

    # --------------------------------------------------------
    # Compact CSV evidence.
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        [
            {
                "acceptance_test": "AC-1",
                "status": ac1_result["status"],
                "leakage_effect_confirmed": (
                    ac1_result[
                        "leakage_effect_confirmed"
                    ]
                ),
                "performance_survives_leakage_removal": (
                    ac1_result[
                        "performance_survives_leakage_removal"
                    ]
                ),
                "all_failures_detected": "",
                "leakage_detected": "",
                "split_integrity_detected": "",
                "representative_transfer_failure_detected": "",
                "unsafe_optimization_detected": "",
                "environment_frozen": "",
                "resources_unchanged": "",
            },

            {
                "acceptance_test": "AC-2",
                "status": ac2_result["status"],
                "leakage_effect_confirmed": "",
                "performance_survives_leakage_removal": "",
                "all_failures_detected": (
                    ac2_result[
                        "all_failures_detected"
                    ]
                ),
                "leakage_detected": (
                    ac2_result[
                        "leakage_detected"
                    ]
                ),
                "split_integrity_detected": (
                    ac2_result[
                        "split_integrity_detected"
                    ]
                ),
                "representative_transfer_failure_detected": (
                    ac2_result[
                        "representative_transfer_failure_detected"
                    ]
                ),
                "unsafe_optimization_detected": (
                    ac2_result[
                        "unsafe_optimization_detected"
                    ]
                ),
                "environment_frozen": "",
                "resources_unchanged": "",
            },

            {
                "acceptance_test": "AC-4",
                "status": ac4_result["status"],
                "leakage_effect_confirmed": "",
                "performance_survives_leakage_removal": "",
                "all_failures_detected": "",
                "leakage_detected": "",
                "split_integrity_detected": "",
                "representative_transfer_failure_detected": "",
                "unsafe_optimization_detected": "",
                "environment_frozen": (
                    ac4_result["environment_frozen"]
                ),
                "resources_unchanged": (
                    ac4_result["resources_unchanged"]
                ),
            },
        ]
    )

    result_df.to_csv(
        ACCEPTANCE_RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Detailed JSON evidence.
    # --------------------------------------------------------

    complete_report = {
        "AC-1": ac1_result,
        "AC-2": ac2_result,
        "AC-4": ac4_result,
    }

    with open(
        ACCEPTANCE_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            complete_report,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Console output.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ACCEPTANCE TEST HARNESS")
    print("=" * 70)

    # --------------------------------------------------------
    # AC-1 output
    # --------------------------------------------------------

    print("\nAC-1 -", ac1_result["status"])

    print(
        "Leakage effect confirmed:",
        ac1_result[
            "leakage_effect_confirmed"
        ],
    )

    print(
        "Performance survives leakage removal:",
        ac1_result[
            "performance_survives_leakage_removal"
        ],
    )

    print("\nMetric differences:")

    for metric, difference in ac1_result[
        "metric_differences"
    ].items():

        print(
            f"  {metric}: {difference:.10f}"
        )

    # --------------------------------------------------------
    # AC-2 output
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("\nAC-2 -", ac2_result["status"])

    print(
        "All mandatory tests pass:",
        ac2_result["all_tests_pass"],
    )

    print(
        "All failures detected:",
        ac2_result["all_failures_detected"],
    )

    print(
        "Leakage detected:",
        ac2_result["leakage_detected"],
    )

    print(
        "Temporal/group split violation detected:",
        ac2_result[
            "split_integrity_detected"
        ],
    )

    print(
        "Representative transfer failure detected:",
        ac2_result[
            "representative_transfer_failure_detected"
        ],
    )

    print(
        "Unsafe optimization detected:",
        ac2_result[
            "unsafe_optimization_detected"
        ],
    )

    print("\nMandatory negative tests:")

    for test_id, test_result in ac2_result[
        "mandatory_negative_tests"
    ].items():

        print(
            f"  {test_id}: "
            f"{test_result['status']} | "
            f"failure_detected="
            f"{test_result['failure_detected']}"
        )

    print("\nSafe response:")
    print(ac2_result["safe_response"])

    # --------------------------------------------------------
    # AC-4 output
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("\nAC-4 -", ac4_result["status"])

    print(
        "Frozen resource envelope:",
        ac4_result["environment_frozen"],
    )

    print(
        "Resources unchanged:",
        ac4_result["resources_unchanged"],
    )

    print("\nSafe response:")
    print(ac4_result["safe_response"])

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"Evidence saved to:\n"
        f"{ACCEPTANCE_RESULTS_FILE}\n"
        f"{ACCEPTANCE_REPORT_FILE}"
    )


if __name__ == "__main__":
    main()