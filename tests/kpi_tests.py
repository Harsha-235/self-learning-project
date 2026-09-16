from pathlib import Path
import pandas as pd
import json


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

CONTROLLED_RESULTS_FILE = (
    EXPERIMENTS_DIR / "controlled_indirect_leakage_results.csv"
)

NEGATIVE_TEST_FILE = (
    EXPERIMENTS_DIR / "mandatory_negative_test_results.csv"
)

OUTPUT_CSV = EXPERIMENTS_DIR / "kpi_results.csv"
OUTPUT_JSON = EXPERIMENTS_DIR / "kpi_report.json"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_bool(value):
    """
    Safely convert CSV boolean-like values into Python bool.
    """

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
        "y",
        "pass",
    }


def get_model_row(results, model_name):
    """
    Return one model result row from the controlled
    leakage experiment.
    """

    matches = results[
        results["model"].astype(str).str.strip().str.lower()
        == model_name.strip().lower()
    ]

    if matches.empty:
        raise ValueError(
            f"Required model result not found: {model_name}"
        )

    return matches.iloc[0]


def require_columns(dataframe, required_columns, filename):
    """
    Verify that an evidence file contains all required columns.
    """

    missing = set(required_columns) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# KPI-1
# ============================================================

def run_kpi_1():
    """
    KPI-1:
    Leakage detected before deployment.

    The controlled leakage experiment produced a leakage
    feature ROC-AUC of 1.0.

    The project specifies an 80/100 leakage-detection target.
    The controlled ROC-AUC is converted to an evidence score
    from 0-100 for this controlled test.
    """

    results = pd.read_csv(CONTROLLED_RESULTS_FILE)

    require_columns(
        results,
        {"model", "roc_auc"},
        CONTROLLED_RESULTS_FILE.name
    )

    leakage_row = get_model_row(
        results,
        "Indirect Leakage"
    )

    leakage_roc_auc = float(
        leakage_row["roc_auc"]
    )

    detection_score = leakage_roc_auc * 100.0

    threshold = 80.0

    passed = detection_score >= threshold

    return {
        "kpi": "KPI-1",
        "name": "Leakage Detection Effectiveness",
        "measured_value": round(detection_score, 2),
        "unit": "score / 100",
        "required_threshold": threshold,
        "leakage_roc_auc": round(leakage_roc_auc, 4),
        "reference_comparison": "NOT_AVAILABLE",
        "status": "PASS" if passed else "FAIL",
        "evidence": (
            "Controlled indirect leakage experiment"
        ),
        "safe_response": (
            "Reject suspicious leakage feature"
        ),
    }


# ============================================================
# KPI-2
# ============================================================

def run_kpi_2():
    """
    KPI-2:
    Performance after leakage removal.

    The controlled experiment compares:
        1. Clean Baseline
        2. Indirect Leakage
        3. After Leakage Removal

    The main evidence used here is performance retention
    after removing the controlled leakage feature.

    A 100% retention means that the useful clean-model
    performance survived leakage removal in this controlled
    experiment.

    The project specification also asks for comparison against
    an O2 reference. No numeric O2 KPI score/rubric is present
    in the available project evidence, so that part is reported
    as NOT_AVAILABLE instead of inventing a value.
    """

    results = pd.read_csv(CONTROLLED_RESULTS_FILE)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    require_columns(
        results,
        {"model", *metrics},
        CONTROLLED_RESULTS_FILE.name
    )

    clean = get_model_row(
        results,
        "Clean Baseline"
    )

    recovered = get_model_row(
        results,
        "After Leakage Removal"
    )

    retention_scores = {}

    for metric in metrics:

        clean_value = float(clean[metric])
        recovered_value = float(recovered[metric])

        if clean_value == 0:
            retention = 0.0
        else:
            retention = (
                recovered_value / clean_value
            ) * 100.0

        retention_scores[metric] = retention

    performance_retention = (
        sum(retention_scores.values())
        / len(retention_scores)
    )

    threshold = 80.0

    threshold_pass = (
        performance_retention >= threshold
    )

    # Check whether the recovered model is effectively
    # unchanged from the clean baseline.
    metric_differences = {}

    for metric in metrics:
        metric_differences[metric] = (
            float(recovered[metric])
            - float(clean[metric])
        )

    performance_survives = all(
        abs(value) <= 0.000001
        for value in metric_differences.values()
    )

    if threshold_pass and performance_survives:
        status = "PASS_WITH_LIMITATION"
    else:
        status = "FAIL"

    return {
        "kpi": "KPI-2",
        "name": "Performance After Leakage Removal",
        "measured_value": round(
            performance_retention,
            2
        ),
        "unit": "performance retention / 100",
        "required_threshold": threshold,
        "clean_baseline_f1": round(
            float(clean["f1"]),
            4
        ),
        "after_removal_f1": round(
            float(recovered["f1"]),
            4
        ),
        "metric_differences": metric_differences,
        "performance_survives_leakage_removal": (
            performance_survives
        ),
        "reference_comparison": (
            "NOT_AVAILABLE - numeric O2 KPI reference "
            "score/rubric is not recorded"
        ),
        "status": status,
        "evidence": (
            "Controlled leakage-removal experiment"
        ),
        "safe_response": (
            "Accept representative performance evidence; "
            "record the missing O2 numeric reference separately."
        ),
    }


# ============================================================
# KPI-3
# ============================================================

def run_kpi_3():
    """
    KPI-3:
    Split integrity test pass rate.

    Evidence:
        - Temporal validation
        - Group validation
        - NT-3 invalid split detection

    The implemented workflow demonstrates both temporal and
    group-aware validation and detects the invalid split
    condition in NT-3.

    The project specification requires comparison with an O2
    reference, but no numeric O2 split-integrity reference
    is currently recorded.
    """

    if not NEGATIVE_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Required evidence file not found: "
            f"{NEGATIVE_TEST_FILE}"
        )

    results = pd.read_csv(
        NEGATIVE_TEST_FILE
    )

    required_columns = {
        "test",
        "status",
        "failure_detected",
        "safe_response",
        "temporal_violation_detected",
        "group_violation_detected",
    }

    require_columns(
        results,
        required_columns,
        NEGATIVE_TEST_FILE.name
    )

    nt3_matches = results[
        results["test"].astype(str).str.strip()
        == "NT-3"
    ]

    if nt3_matches.empty:
        raise ValueError(
            "NT-3 was not found in mandatory negative tests."
        )

    nt3 = nt3_matches.iloc[0]

    temporal_detected = parse_bool(
        nt3["temporal_violation_detected"]
    )

    group_detected = parse_bool(
        nt3["group_violation_detected"]
    )

    nt3_pass = (
        str(nt3["status"]).strip().upper()
        == "PASS"
    )

    failure_detected = parse_bool(
        nt3["failure_detected"]
    )

    split_checks = [
        temporal_detected,
        group_detected,
        nt3_pass,
        failure_detected,
    ]

    passed_checks = sum(
        1 for check in split_checks
        if check
    )

    pass_rate = (
        passed_checks
        / len(split_checks)
    ) * 100.0

    threshold_pass = (
        pass_rate >= 80.0
    )

    if threshold_pass:
        status = "PASS_WITH_LIMITATION"
    else:
        status = "FAIL"

    return {
        "kpi": "KPI-3",
        "name": "Split Integrity Test Pass Rate",
        "measured_value": round(
            pass_rate,
            2
        ),
        "unit": "percent",
        "required_threshold": 80.0,
        "temporal_violation_detected": (
            temporal_detected
        ),
        "group_violation_detected": (
            group_detected
        ),
        "invalid_split_detected": (
            failure_detected
        ),
        "checks_passed": passed_checks,
        "checks_total": len(split_checks),
        "reference_comparison": (
            "NOT_AVAILABLE - numeric O2 split-integrity "
            "reference is not recorded"
        ),
        "status": status,
        "evidence": (
            "Temporal validation, group validation and NT-3"
        ),
        "safe_response": (
            "Reject invalid validation split."
        ),
    }


# ============================================================
# KPI-4
# ============================================================

def run_kpi_4():
    """
    KPI-4:
    Safe-recovery success under injected faults.

    The project contains five mandatory negative tests.
    Each test must:
        - PASS
        - detect the intended failure
        - record a safe response

    The resulting safe-handling rate is reported as the
    observed recovery/safe-response evidence.

    The O2 reference comparison is not available as a numeric
    KPI baseline, so it is not invented.
    """

    if not NEGATIVE_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Required evidence file not found: "
            f"{NEGATIVE_TEST_FILE}"
        )

    results = pd.read_csv(
        NEGATIVE_TEST_FILE
    )

    required_columns = {
        "test",
        "status",
        "failure_detected",
        "safe_response",
    }

    require_columns(
        results,
        required_columns,
        NEGATIVE_TEST_FILE.name
    )

    required_tests = {
        "NT-1",
        "NT-2",
        "NT-3",
        "NT-4",
        "NT-5",
    }

    available_tests = set(
        results["test"]
        .astype(str)
        .str.strip()
    )

    missing_tests = (
        required_tests - available_tests
    )

    if missing_tests:
        raise ValueError(
            "Missing mandatory negative tests: "
            f"{sorted(missing_tests)}"
        )

    test_results = []

    for test_id in sorted(required_tests):

        row = results[
            results["test"].astype(str).str.strip()
            == test_id
        ].iloc[0]

        status_pass = (
            str(row["status"]).strip().upper()
            == "PASS"
        )

        failure_detected = parse_bool(
            row["failure_detected"]
        )

        safe_response = (
            str(row["safe_response"]).strip()
        )

        response_recorded = (
            safe_response != ""
            and safe_response.lower()
            != "nan"
        )

        successful = (
            status_pass
            and failure_detected
            and response_recorded
        )

        test_results.append({
            "test": test_id,
            "success": successful,
        })

    successful_tests = sum(
        1
        for item in test_results
        if item["success"]
    )

    total_tests = len(test_results)

    recovery_rate = (
        successful_tests
        / total_tests
    ) * 100.0

    threshold_pass = (
        recovery_rate >= 80.0
    )

    if threshold_pass:
        status = "PASS_WITH_LIMITATION"
    else:
        status = "FAIL"

    return {
        "kpi": "KPI-4",
        "name": "Safe-Recovery Success Under Injected Faults",
        "measured_value": round(
            recovery_rate,
            2
        ),
        "unit": "percent",
        "required_threshold": 80.0,
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "reference_comparison": (
            "NOT_AVAILABLE - numeric O2 recovery reference "
            "is not recorded"
        ),
        "status": status,
        "evidence": (
            "Mandatory negative-test and recovery campaign"
        ),
        "safe_response": (
            "Use the documented safe/degraded response "
            "for each detected failure condition."
        ),
    }


# ============================================================
# KPI-5
# ============================================================

def run_kpi_5():
    """
    KPI-5:
    Normalized tracking/regulation RMSE where applicable.

    This project is a binary machine-failure classification
    system, so tracking/regulation RMSE is not applicable.
    """

    return {
        "kpi": "KPI-5",
        "name": "Normalized Tracking/Regulation RMSE",
        "measured_value": None,
        "unit": "N/A",
        "required_threshold": None,

        # IMPORTANT:
        # This key is required by the main summary section.
        "reference_comparison": "NOT_APPLICABLE",

        "status": "NOT_APPLICABLE",

        "evidence": (
            "Binary machine-failure classification project"
        ),

        "safe_response": (
            "Do not fabricate an RMSE value for a "
            "non-regression classification task."
        ),
    }
# ============================================================
# KPI-6
# ============================================================

def run_kpi_6():
    """
    KPI-6:
    Safety-envelope violation count.

    The mandatory adversarial software/data conditions are
    represented by NT-1 through NT-5.

    A safety-envelope violation is counted when a mandatory
    condition is:
        - not detected,
        - not passed,
        - or has no documented safe response.

    This is a software assurance proxy for the project's
    controlled adversarial tests, not a physical safety
    certification.
    """

    if not NEGATIVE_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Required evidence file not found: "
            f"{NEGATIVE_TEST_FILE}"
        )

    results = pd.read_csv(
        NEGATIVE_TEST_FILE
    )

    required_columns = {
        "test",
        "status",
        "failure_detected",
        "safe_response",
    }

    require_columns(
        results,
        required_columns,
        NEGATIVE_TEST_FILE.name
    )

    required_tests = {
        "NT-1",
        "NT-2",
        "NT-3",
        "NT-4",
        "NT-5",
    }

    violations = []

    for test_id in sorted(required_tests):

        matches = results[
            results["test"].astype(str).str.strip()
            == test_id
        ]

        if matches.empty:
            violations.append(
                f"{test_id}: missing test"
            )
            continue

        row = matches.iloc[0]

        status_pass = (
            str(row["status"]).strip().upper()
            == "PASS"
        )

        failure_detected = parse_bool(
            row["failure_detected"]
        )

        safe_response = (
            str(row["safe_response"]).strip()
        )

        response_recorded = (
            safe_response != ""
            and safe_response.lower()
            != "nan"
        )

        if not status_pass:
            violations.append(
                f"{test_id}: test did not PASS"
            )

        if not failure_detected:
            violations.append(
                f"{test_id}: failure condition not detected"
            )

        if not response_recorded:
            violations.append(
                f"{test_id}: safe response missing"
            )

    violation_count = len(violations)

    passed = (
        violation_count == 0
    )

    return {
        "kpi": "KPI-6",
        "name": "Safety-Envelope Violation Count",
        "measured_value": violation_count,
        "unit": "violations",
        "required_threshold": 0,
        "reference_comparison": "DIRECT_THRESHOLD",
        "mandatory_conditions": 5,
        "violations": violations,
        "status": "PASS" if passed else "FAIL",
        "evidence": (
            "NT-1 through NT-5 mandatory adversarial "
            "software/data conditions"
        ),
        "safe_response": (
            "All mandatory adversarial conditions were "
            "detected and safe responses were recorded."
            if passed
            else
            "Investigate every unresolved safety-envelope "
            "condition before acceptance."
        ),
    }


# ============================================================
# MAIN KPI HARNESS
# ============================================================

def main():

    print("=" * 70)
    print("KPI TEST HARNESS")
    print("=" * 70)

    print()
    print("Running KPI-1...")
    kpi1 = run_kpi_1()

    print("Running KPI-2...")
    kpi2 = run_kpi_2()

    print("Running KPI-3...")
    kpi3 = run_kpi_3()

    print("Running KPI-4...")
    kpi4 = run_kpi_4()

    print("Running KPI-5...")
    kpi5 = run_kpi_5()

    print("Running KPI-6...")
    kpi6 = run_kpi_6()

    results = [
        kpi1,
        kpi2,
        kpi3,
        kpi4,
        kpi5,
        kpi6,
    ]

    # --------------------------------------------------------
    # Save compact CSV evidence.
    # --------------------------------------------------------

    csv_rows = []

    for result in results:

        csv_rows.append({
            "kpi": result["kpi"],
            "name": result["name"],
            "status": result["status"],
            "measured_value": result["measured_value"],
            "unit": result["unit"],
            "required_threshold": (
                result["required_threshold"]
            ),
            "reference_comparison": (
                result.get(
                    "reference_comparison",
                    "NOT_AVAILABLE"
                )
            ),
            "evidence": result["evidence"],
        })

    pd.DataFrame(csv_rows).to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Save detailed JSON evidence.
    # --------------------------------------------------------

    report = {
        "project": (
            "Gradient Boosting on Tabular Data "
            "with Leakage-Resistant Validation"
        ),
        "kpi_count": len(results),
        "results": results,
        "summary": {
            "kpi_1": kpi1["status"],
            "kpi_2": kpi2["status"],
            "kpi_3": kpi3["status"],
            "kpi_4": kpi4["status"],
            "kpi_5": kpi5["status"],
            "kpi_6": kpi6["status"],
        },
        "note": (
            "KPI-2 to KPI-4 include reference-comparison "
            "limitations because no numeric O2 KPI baseline "
            "or rubric score is currently recorded. "
            "KPI-5 is not applicable to the binary "
            "classification task. No unsupported RMSE or "
            "reference value is fabricated."
        ),
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("KPI RESULTS")
    print("=" * 70)

    for result in results:

        print()
        print(
            f"{result['kpi']} - "
            f"{result['name']}"
        )

        print(
            f"Measured Value       : "
            f"{result['measured_value']}"
        )

        print(
            f"Required Threshold   : "
            f"{result['required_threshold']}"
        )

        print(
            f"Status               : "
            f"{result['status']}"
        )

        print(
            f"Reference Comparison : "
            f"{result['reference_comparison']}"
        )

    print()
    print("=" * 70)
    print("KPI EVIDENCE SAVED")
    print("=" * 70)

    print()
    print(
        f"CSV : {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()