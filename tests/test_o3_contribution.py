"""
Unit Tests and Verification Oracle for Deliverable D3 (O3 Contribution)
"""

from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
O3_ABLATION_CSV = PROJECT_ROOT / "experiments" / "o3_ablation_results.csv"
O3_SENSITIVITY_CSV = PROJECT_ROOT / "experiments" / "o3_sensitivity_results.csv"
O3_COMPARISON_CSV = PROJECT_ROOT / "experiments" / "o3_candidate_vs_reference.csv"
O3_REPORT_JSON = PROJECT_ROOT / "experiments" / "o3_contribution_report.json"


def test_o3_artifacts_exist():
    assert O3_ABLATION_CSV.exists(), "O3 ablation CSV missing."
    assert O3_SENSITIVITY_CSV.exists(), "O3 sensitivity CSV missing."
    assert O3_COMPARISON_CSV.exists(), "O3 candidate-vs-reference CSV missing."
    assert O3_REPORT_JSON.exists(), "O3 contribution report JSON missing."


def test_o3_ablation_progression():
    df = pd.read_csv(O3_ABLATION_CSV)
    assert len(df) == 5, "Expected 5 ablation configurations."
    # The full ensemble must score higher than single methods
    full_score = df.loc[df["configuration"] == "Full_O3_Ensemble", "rubric_score"].iloc[0]
    single_score = df.loc[df["configuration"] == "MDI_Baseline_Only", "rubric_score"].iloc[0]
    assert full_score > single_score, "Full ensemble must outperform baseline MDI."
    assert full_score >= 80.0, "Full ensemble must meet the >= 80 rubric threshold."


def test_o3_sensitivity_trend():
    df = pd.read_csv(O3_SENSITIVITY_CSV)
    assert len(df) >= 5, "Expected at least 5 noise level evaluations."
    # Low noise must achieve 100% detection rate
    assert df.loc[df["noise_sigma"] == 0.01, "detection_rate_pct"].iloc[0] == 100.0


def test_o3_candidate_superiority():
    with open(O3_REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["status"] == "APPROVED"
    assert report["meets_kpi_target"] is True
    assert report["improvement_points"] >= 5.0, "O3 must improve over O2 by at least 5 points."
    assert report["candidate_rubric_score"] >= 80.0, "O3 candidate must score >= 80."


if __name__ == "__main__":
    test_o3_artifacts_exist()
    test_o3_ablation_progression()
    test_o3_sensitivity_trend()
    test_o3_candidate_superiority()
    print("All O3 Contribution Oracle tests PASS!")
