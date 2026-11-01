# Deliverable D1: Source-to-Claim Traceability Matrix

This document provides a line-item trace from every problem statement requirement, functional/non-functional requirement, acceptance condition, KPI target, and negative test to its concrete implementation, test oracle, and evidence artifact.

---

### 1. Objectives & Milestones Traceability

| ID | Description | Primary Implementation | Test Oracle / Fixture | Evidence Artifact | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **O1** | Tabular prediction baseline with Gradient Boosting | `ml/baseline_model.py`, `ml/model_pipeline.py` | `tests/acceptance_tests.py` | `models/gradient_boosting_model.pkl`, `experiments/final_model_results.csv` | Approved (Gate 1) |
| **O2** | Feature importance reference (compare 2 methods, fixtures, oracle, resource profile) | `ml/feature_importance_reference.py` | `tests/test_o2_reference.py` | `experiments/o2_reference_results.csv`, `experiments/o2_resource_profile.json` | Approved (Gate 2) |
| **O3** | Indirect target encoding detection (ablation, sensitivity, baseline-vs-candidate) | `ml/indirect_leakage_detector.py`, `ml/indirect_leakage_experiment.py` | `tests/test_o3_contribution.py` | `experiments/o3_ablation_results.csv`, `experiments/o3_sensitivity_results.csv`, `experiments/controlled_indirect_leakage_results.csv` | Approved (Gate 3) |
| **O4** | Temporal and group separation in validation splits with operable telemetry and recovery | `ml/split_integrity_verifier.py`, `ml/safety_interlock.py`, `backend/app.py` | `tests/negative_tests.py`, `tests/test_split_integrity.py` | `experiments/validation_results.csv`, `experiments/mandatory_negative_test_log.json` | Approved (Gate 4) |
| **O5** | Independent acceptance validation surviving leakage removal across AC-1..AC-4 and NT-1..NT-5 | `tests/acceptance_tests.py`, `tests/kpi_tests.py`, `tests/negative_tests.py` | `tests/independent_acceptance_test.py` | `experiments/acceptance_results.csv`, `experiments/kpi_results.csv`, `experiments/acceptance_report.json` | Final Gate (Passed) |

---

### 2. Functional & Non-Functional Requirements Traceability

| Req ID | Requirement Statement | Implementation File | Verification Test |
| :--- | :--- | :--- | :--- |
| **FR-1** | Ingest tabular datasets with natural/injected leakage; validate schema & quality gates | `ml/data_loader.py`, `main.py` | Data validation in `tests/acceptance_tests.py` |
| **FR-2** | Reproduce O2 feature importance via one-command reference path with independent oracle | `ml/feature_importance_reference.py`, `main.py` | `python main.py --mode reference` |
| **FR-3** | Implement O3 indirect target encoding detector as testable, versioned subsystem | `ml/indirect_leakage_detector.py` | `tests/test_o3_contribution.py` |
| **FR-4** | Implement & verify temporal/group validation splits with telemetry contracts | `ml/split_integrity_verifier.py` | `tests/negative_tests.py` (NT-3) |
| **FR-5** | Implement & verify Gradient Boosting with input/output error & telemetry contracts | `ml/model_pipeline.py`, `backend/app.py` | `tests/acceptance_tests.py` (AC-1) |
| **FR-6** | Deliver complete demonstrable system (no isolated notebooks or dashboards alone) | Full stack: `ml/`, `backend/`, `frontend/`, `main.py` | Integrated acceptance harness |
| **FR-7** | Detect NT-1 condition, invoke safe/degraded response, retain recovery evidence | `ml/safety_interlock.py`, `tests/negative_tests.py` | `tests/negative_tests.py` (NT-1) |
| **NFR-1** | Performance: Meet all KPI targets under AC-1 and AC-4 | `tests/kpi_tests.py` | KPI evaluation suite |
| **NFR-2** | Reliability & Recovery: Pass every negative test without hiding failures in averages | `tests/negative_tests.py` | 5/5 negative test suite |
| **NFR-3** | Domain assurance: Explicit sensor models, timing, actuation, and safety limits | `docs/D1_ENGINEERING_CHARTER_AND_SPECIFICATION.md` | Boundary test fixtures |
| **NFR-4** | Evidence quality: Correlated physical simulator/measured data | `data/machine_data_v2.csv` | Dataset provenance manifest |
| **NFR-5** | Maintainability: Automated tests, documented interfaces, clean environment build | `requirements.txt`, `tests/`, `main.py` | `pytest` / test runner |
| **NFR-6** | Operability: Logs, telemetry, health indicators, degraded mode, recovery procedure | `backend/app.py`, `ml/safety_interlock.py` | Health & status API endpoints |

---

### 3. Acceptance Conditions (AC) Traceability

| AC ID | Name | Standard / Verification Method | Pass Rule | Status |
| :--- | :--- | :--- | :--- | :--- |
| **AC-1** | Representative Operation | Performance retention after leakage feature quarantine | Retention $\ge 80\%$ and clean-vs-recovered metric delta $\le 10^{-6}$ | PASS |
| **AC-2** | Boundary & Failure Operation | Execution of mandatory negative test campaign (NT-1 to NT-5) | 5/5 failure conditions detected with logged safe responses | PASS |
| **AC-3** | Independent Acceptance Evidence | End-to-end repeated execution using independent test oracle | Complete reproduction of detection, quarantine, and zero drift | PASS |
| **AC-4** | Frozen Resource Envelope | Hardware, software versions, workload, and random seeds verified | Zero unapproved changes between baseline and candidate | PASS |

---

### 4. KPI Contract Traceability

| KPI ID | Measure | Baseline (O2) | Direction | Unit | Target / Pass Rule | Guardrail |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KPI-1** | Leakage detected before deployment | 42.0 (MDI alone) | Higher $\uparrow$ | 0–100 Rubric | $\ge 80/100$ and $\ge \text{O2} + 5\text{ pts}$ | No safety-envelope violations |
| **KPI-2** | Performance after removal | 77.9 (Baseline AUC) | Higher $\uparrow$ | 0–100 Rubric | $\ge 80/100$ and $\ge \text{O2} + 5\text{ pts}$ | No safety-envelope violations |
| **KPI-3** | Split integrity test pass rate | 92.0% | Higher $\uparrow$ | % | $\ge \text{O2} + 3\%$ ($100.0\%$) | No temporal/group contamination |
| **KPI-4** | Safe-recovery success rate under faults | 90.0% | Higher $\uparrow$ | % | $\ge \text{O2} + 3\%$ ($100.0\%$) | Bounded recovery $< 500\text{ ms}$ |
| **KPI-5** | Operating trajectory tracking RMSE | 14.8% | Lower $\downarrow$ | % of range | $\le 0.80 \times \text{O2}$ ($\le 11.8\%$) | Calibrated probability alignment |
| **KPI-6** | Safety-envelope violation count | 0 | Lower $\downarrow$ | Count | **Strictly $= 0$** | Zero tolerance across all tests |

---

### 5. Mandatory Negative-Test Campaign Traceability

| Test ID | Trigger Scenario | Expected Safe Behavior | Recovery Protocol | Evidence File |
| :--- | :--- | :--- | :--- | :--- |
| **NT-1** | Validation score collapses upon deployment temporal shift ($\ge 30\%$ drop) | Trigger `COLLAPSE_INTERLOCK`, hold deployment | Revert to conservative reference, re-estimate boundary | `mandatory_negative_test_results.csv` |
| **NT-2** | Target-derived proxy leakage exploited by GBDT | Trigger `LEAKAGE_INTERLOCK`, isolate feature | Quarantine feature, retrain clean model | `mandatory_negative_test_results.csv` |
| **NT-3** | Split violates temporal ordering or leaks machine IDs | Trigger `SPLIT_VIOLATION_INTERLOCK`, reject split | Re-partition data with strict temporal/group disjoint boundary | `mandatory_negative_test_results.csv` |
| **NT-4** | High simulation performance does not transfer to representative data | Trigger `DOMAIN_SHIFT_INTERLOCK`, enter degraded mode | Re-calibrate thresholds using representative sample | `mandatory_negative_test_results.csv` |
| **NT-5** | Nominal optimization improves average accuracy but increases tail risk | Trigger `RISK_INTERLOCK`, reject optimization | Revert to conservative baseline with bounded tail risk | `mandatory_negative_test_results.csv` |
