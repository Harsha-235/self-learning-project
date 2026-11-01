"""
Unit Tests and Verification Oracle for Deliverable D2 (O2 Reference)
"""

from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
O2_RESULTS_CSV = PROJECT_ROOT / "experiments" / "o2_reference_results.csv"
O2_COMPARISON_CSV = PROJECT_ROOT / "experiments" / "o2_feature_importance_comparison.csv"
O2_PROFILE_JSON = PROJECT_ROOT / "experiments" / "o2_resource_profile.json"


def test_o2_artifacts_exist():
    assert O2_RESULTS_CSV.exists(), "O2 reference results CSV missing."
    assert O2_COMPARISON_CSV.exists(), "O2 feature importance comparison CSV missing."
    assert O2_PROFILE_JSON.exists(), "O2 resource profile JSON missing."


def test_o2_both_approaches_computed():
    df = pd.read_csv(O2_COMPARISON_CSV)
    required_cols = {"feature", "mdi_importance", "permutation_importance"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns in {O2_COMPARISON_CSV}"
    assert len(df) == 7, "Expected 7 legitimate sensor features."
    assert (df["mdi_importance"] >= 0).all(), "MDI importances must be non-negative."
    assert (df["mdi_importance"].sum() - 1.0) < 1e-4, "MDI importances should sum to 1.0."


def test_o2_resource_profile():
    with open(O2_PROFILE_JSON, "r", encoding="utf-8") as f:
        profile = json.load(f)
    assert "total_runtime_seconds" in profile
    assert "peak_memory_mb" in profile
    assert profile["total_runtime_seconds"] > 0
    assert profile["peak_memory_mb"] > 0
    assert "selected_reference" in profile


def test_o2_kpi_baselines_frozen():
    df = pd.read_csv(O2_RESULTS_CSV)
    expected_kpis = {"KPI-1", "KPI-2", "KPI-3", "KPI-4", "KPI-5", "KPI-6"}
    assert set(df["kpi_id"]) == expected_kpis, "All 6 KPIs must have frozen O2 baselines."
    assert (df["o2_baseline_value"] >= 0).all(), "Baseline values must be non-negative."


if __name__ == "__main__":
    test_o2_artifacts_exist()
    test_o2_both_approaches_computed()
    test_o2_resource_profile()
    test_o2_kpi_baselines_frozen()
    print("All O2 Reference Oracle tests PASS!")
