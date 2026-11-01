# Gradient Boosting on Tabular Data With Leakage-Resistant Validation
## Comprehensive Final Engineering Project Report (CP1 + CP2)

---

### Executive Abstract
Gradient Boosted Decision Trees (GBDT) dominate tabular machine learning. However, tree ensembles excel at aggressively exploiting data leakage—specifically features that indirectly or directly proxy the target variable, or validation folds that violate temporal ordering and group integrity. In conventional evaluation workflows, leaky models achieve deceptively high validation scores (e.g., ROC-AUC approaching 1.00) that abruptly collapse upon deployment under operational domain shifts.

This capstone project engineered an operable, leakage-resistant machine learning and validation system for industrial condition monitoring and machine failure prediction. The system implements:
1. A reproducible **O2 Reference Benchmark** comparing two distinct feature importance approaches (Mean Decrease in Impurity vs. Permutation Importance).
2. A novel **O3 Indirect Target-Encoding Leakage Detector** combining non-linear mutual information, solo proxy reconstruction AUC, shadow feature benchmarks, and a 5-criteria 0–100 rubric.
3. An **O4 Validation Split Integrity Verifier** enforcing strict chronological ordering and entity disjointness.
4. A **Closed-Loop Safety Interlock Controller** with state machine transitions, degraded conservative fallback, and bounded recovery verification.
5. An **Automated Master Acceptance Harness** proving that performance survives leakage removal across all Acceptance Conditions (**AC-1 to AC-4**), all Key Performance Indicators (**KPI-1 to KPI-6**), and all Mandatory Negative Tests (**NT-1 to NT-5**), with **zero safety envelope violations**.

---

## 1. Problem Statement & Operational Context

Industrial equipment reliability relies on timely, calibrated condition-based maintenance forecasting. When predictive models are trained on sensor records containing:
- Post-failure indicators or maintenance records leaking the target variable,
- Future temporal observations contaminating earlier training steps, or
- Cross-asset contamination where the model memorizes specific machine signatures rather than general physical degradation mechanics,

the models appear flawless during offline validation but produce catastrophic false negatives in deployment. The operational consequence is unplanned machine trip, secondary mechanical damage, or unhandled safety envelope breaches.

The engineering mandate frozen in Gate 1 requires:
> **"Deliver tabular modelling with leakage detected before deployment rather than diagnosed after it, and verify that performance survives leakage removal under representative, boundary, and failure conditions."**

---

## 2. System Architecture & Engineering Contracts

### 2.1 Physical Plant & Sensor Specifications

The system is deployed on multi-sensor industrial rotating machinery operating in discrete 1-hour sampling intervals ($\Delta t = 1.0\text{ h}$):

| Sensor / Signal | Physical Role | Units | Nominal Range | Warning Limit | Safety Trip Limit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Temperature** | Bearing/casing thermal dissipation | °C | 50.0 – 75.0 | > 85.0 | > 110.0 |
| **Pressure** | Hydraulic line pressure | kPa | 95.0 – 105.0 | > 115.0 | > 130.0 |
| **Vibration** | Tri-axial RMS vibration velocity | mm/s | 0.50 – 2.00 | > 3.50 | > 5.00 |
| **Rotational Speed** | Main shaft tachometer | RPM | 1200 – 1800 | < 1000 or > 2000 | < 800 or > 2200 |
| **Operating Hours** | Monotonic asset run-time | Hours | 0 – 10,000 | > 8,000 | > 12,000 |
| **Ambient Temp** | Environmental intake sensor | °C | 15.0 – 35.0 | > 40.0 | > 50.0 |
| **Load Percent** | Mechanical load fraction | % | 40.0 – 80.0 | > 90.0 | > 100.0 |

* **Target Variable**: Binary failure state $y \in \{0, 1\}$.
* **Workload**: 5,000 observations across 10 machines (`M01` to `M10`).

### 2.2 Closed-Loop Safety Interlock Architecture

The system enforces fail-safe operation using an asynchronous closed-loop safety controller (`ml/safety_interlock.py`):

```text
       [ Nominal Operating Stream ]
                     │
                     ▼
       ┌───────────────────────────┐
       │   Sensor Envelope Check   │ ──(Exceeds Limit)──► [ TRIP INTERLOCK ]
       └───────────────────────────┘                             │
                     │ (Within Bounds)                           ▼
                     ▼                                 ┌──────────────────┐
       ┌───────────────────────────┐                   │  DEGRADED MODE   │
       │   Split Integrity Check   │                   │ (Conservative    │
       │  (Temporal / Group Folds) │ ──(Contamination)─► Physical Fallback│
       └───────────────────────────┘                   │  Heuristic Prob) │
                     │ (Verified Valid)                └──────────────────┘
                     ▼                                           │
       ┌───────────────────────────┐                             │
       │ O3 Indirect Leakage Scan  │ ──(Leaky Proxy)───► [ QUARANTINE FEATURE ]
       └───────────────────────────┘                             │
                     │ (Legitimate Features)                     ▼
                     ▼                                 ┌──────────────────┐
       ┌───────────────────────────┐                   │ BOUNDED RECOVERY │
       │ Leakage-Resistant GBDT    │                   │ (< 500 ms Latency│
       │ Inference & Telemetry     │                   │  Oracle Verif.)  │
       └───────────────────────────┘                   └──────────────────┘
```

---

## 3. Milestone Deliverables Summary (D1 through D7)

* **D1 — Engineering Charter & Traceability**: Formalized in `docs/D1_ENGINEERING_CHARTER_AND_SPECIFICATION.md`, `docs/SOURCE_TO_CLAIM_MAP.md`, `docs/ARCHITECTURE_DECISION_RECORD.md`, and `docs/RISK_REGISTER.md`.
* **D2 — Reproducible Reference Benchmark (O2)**: Implemented in `ml/feature_importance_reference.py`, with unit test oracle `tests/test_o2_reference.py`, resource profiling in `experiments/o2_resource_profile.json`, and frozen baselines in `experiments/o2_reference_results.csv`.
* **D3 — Differentiating Contribution Subsystem (O3)**: Implemented in `ml/indirect_leakage_detector.py`, with full ablation and sensitivity studies in `experiments/o3_ablation_results.csv` and `experiments/o3_sensitivity_results.csv`.
* **D4 — Operable System Prototype**: Implemented across `ml/split_integrity_verifier.py`, `ml/safety_interlock.py`, enhanced FastAPI service in `backend/app.py`, and interactive dashboard in `frontend/`.
* **D5 — Master Acceptance Harness**: Implemented in `tests/acceptance_tests.py`, `tests/negative_tests.py`, `tests/kpi_tests.py`, and unified runner `tests/run_all_tests.py`.
* **D6 — One-Command Reproducibility Protocol**: Orchestrated in `main.py` supporting `--mode all`, `--mode reference`, `--mode contribution`, `--mode split`, `--mode test`, and `--mode serve`.
* **D7 — Comprehensive Final Validation Report & Viva Guide**: Formalized in `docs/FINAL_PROJECT_REPORT.md` and `docs/PANEL_DEFENSE_VIVA_GUIDE.md`.

---

## 4. O2 Reference Benchmark: Feature Importance & Trade-offs

CP1 Gate 2 mandated comparing at least two feasible reference approaches for feature importance.

### 4.1 Comparative Evaluation

| Feature | Approach 1: MDI (Gini Gain) | Approach 2: Permutation Importance (Holdout AUC Drop) | Variance (Std) | Physical Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **vibration** | 0.3494 | **0.0451** | ±0.0104 | Primary mechanical degradation indicator |
| **temperature** | 0.1489 | **0.0187** | ±0.0045 | Thermal friction / heat buildup |
| **operating_hours** | 0.3281 | **0.0173** | ±0.0066 | Cumulative wear accumulation |
| **load_percent** | 0.0874 | **0.0113** | ±0.0047 | Operational dynamic stress |
| **pressure** | 0.0348 | **0.0011** | ±0.0015 | Hydraulic fluid stability |
| **ambient_temperature** | 0.0322 | **0.0010** | ±0.0016 | Environmental baseline |
| **rotational_speed** | 0.0193 | **0.0000** | ±0.0015 | Controlled operating setpoint |

### 4.2 Documented Reference Trade-Off Analysis
* **Approach 1 (MDI)**: Evaluates in $0.0003\text{ s}$ but displays known vulnerability to continuous cardinality bias and assigns high importance to collinear leakage features without verifying generalization impact.
* **Approach 2 (Permutation Importance)**: Evaluates in $0.0210\text{ s}$ by measuring true holdout AUC degradation under feature shuffling. It reliably separates real predictive signal from collinear noise.
* **Decision**: Permutation Importance was selected as the authoritative O2 reference standard.

### 4.3 Frozen O2 Reference Baselines
The O2 baseline values frozen for KPI benchmarking are:
* **KPI-1 (Leakage Detection Rubric)**: **42.00 / 100** (Indeterminate baseline)
* **KPI-2 (Performance After Removal)**: **77.96 / 100** (Baseline Clean AUC)
* **KPI-3 (Split Integrity Pass Rate)**: **92.00%**
* **KPI-4 (Safe Recovery Success Rate)**: **90.00%**
* **KPI-5 (Normalised Tracking RMSE)**: **4.14%** of operating range
* **KPI-6 (Safety-Envelope Violations)**: **0 count**

---

## 5. O3 Differentiating Contribution: Indirect Target-Encoding Detector

The core contribution is a multi-signal detection subsystem (`ml/indirect_leakage_detector.py`) that evaluates candidate features before model deployment.

### 5.1 5-Criteria 0–100 Rubric Specification
1. **Mutual Information (MI)** ($0–25\text{ pts}$): Non-linear information sharing independent of monotonicity.
2. **Solo Low-Capacity Proxy AUC** ($0–25\text{ pts}$): Prediction power of a depth-2 decision tree on the feature alone.
3. **Shadow Feature Dominance** ($0–20\text{ pts}$): Importance ratio relative to randomized shadow permutations.
4. **Tree Split Priority** ($0–15\text{ pts}$): Frequency of appearing in root split positions.
5. **Correlation Anchor** ($0–15\text{ pts}$): High target separation anchor.

### 5.2 Ablation Study Results

| Configuration | Components Active | Rubric Score | Leaky Feature Flagged? | Defensibility |
| :--- | :--- | :---: | :---: | :--- |
| **MDI Baseline (O2)** | Split impurity alone | 15.0 | ❌ NO | Cannot separate high utility from proxy leak |
| **Solo AUC Only** | Depth-2 tree AUC alone | 33.0 | ❌ NO | Ineffective under subtle noise ($\sigma > 0.15$) |
| **Mutual Info Only** | Non-linear entropy alone | 33.0 | ❌ NO | Vulnerable to false positives on legitimate drivers |
| **MI + Proxy AUC** | Dual-signal test | 65.0 | ⚠️ BORDERLINE | Detects high-signal leaks, misses subtle proxies |
| **Full O3 Ensemble** | **MI + AUC + Shadows + Split Priority** | **100.0** | **✅ YES (QUARANTINED)** | **Full Panel Standard (KPI-1 PASS)** |

### 5.3 Sensitivity Analysis across Noise Envelopes ($\sigma$)

| Injected Noise ($\sigma$) | Mean Rubric Score | Score Std ($\pm$) | Mean Proxy AUC | Detection Rate (%) | Subsystem Behavior |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.01** | 100.0 | 0.00 | 1.0000 | **100.0%** | Defensible immediate quarantine |
| **0.05** | 100.0 | 0.00 | 1.0000 | **100.0%** | Defensible immediate quarantine |
| **0.10** | 100.0 | 0.00 | 0.9996 | **100.0%** | Defensible immediate quarantine |
| **0.20** | 96.0 | 2.00 | 0.9773 | **100.0%** | Defensible immediate quarantine |
| **0.35** | 69.8 | 3.92 | 0.8857 | **100.0%** | Exposed and flagged |
| **0.50** | 52.8 | 3.43 | 0.8014 | **0.0%** | Graceful degradation (noise overwhelms leak) |

---

## 6. O4 Operable System Integration & Split Verification

### 6.1 Split Integrity Verification
The verifier (`ml/split_integrity_verifier.py`) enforces mathematical disjointness:
* **Temporal Disjointness**: $\max(t_{\text{train}}) = \text{2025-01-17 15:00:00} < \min(t_{\text{test}}) = \text{2025-01-17 16:00:00}$ ($\Delta t = 1.0\text{ h}$). Token: `TOKEN-SPLIT-VALID-103130`.
* **Group Disjointness**: $\text{Train}_{\text{machines}} = \{M01..M08\}$, $\text{Test}_{\text{machines}} = \{M09, M10\}$. Intersection $\emptyset$. Token: `TOKEN-SPLIT-VALID-536392`.

### 6.2 Full-Stack Operable Prototype
* **Backend**: FastAPI 2.0 service providing `/predict`, `/system/status`, `/interlock/trip`, `/interlock/recover`, `/telemetry`, and `/leakage/analyze`.
* **Frontend**: Interactive engineering workstation (`frontend/index.html`) featuring real-time sensor preset injection, gauge visualization, live interlock latching, and telemetry event streaming.

---

## 7. Mandatory Negative-Test Campaign Results (NT-1 through NT-5)

All 5 negative test fixtures were executed against the closed-loop safety interlock controller:

| Test ID | Trigger Scenario | Failure Detected? | Interlock Latch | Degraded Mode | Recovery Verified? | Recovery Latency | Final Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NT-1** | Validation collapse in deployment ($\Delta \ge 30\%$) | **YES** | **YES** | **YES** | **YES** | **0.007 ms** | **PASS** |
| **NT-2** | Invisible indirect leakage exploitation | **YES** | **YES** | **YES** | **YES** | **0.006 ms** | **PASS** |
| **NT-3** | Temporal & group split boundary violation | **YES** | **YES** | **YES** | **YES** | **0.004 ms** | **PASS** |
| **NT-4** | Simulation transfer failure (Sim-to-Real) | **YES** | **YES** | **YES** | **YES** | **0.004 ms** | **PASS** |
| **NT-5** | Nominal optimization worsening tail risk | **YES** | **YES** | **YES** | **YES** | **0.003 ms** | **PASS** |

**Summary**: **5 / 5 Mandatory Negative Tests PASSED (100%)**. All recovery times are strictly bounded ($< 0.01\text{ ms} \ll 500\text{ ms}$).

---

## 8. Final KPI Scorecard & Contract Verification

| KPI ID | Measure | Baseline (O2) | Achieved (Candidate) | Target Pass Rule | Delta vs O2 Baseline | Status | Guardrail Verification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **KPI-1** | Leakage Detected Before Deployment | 42.00 | **100.00** | $\ge 80.0$ and $\ge \text{O2} + 5.0$ | **+58.00 pts** | **PASS** | Maintained (0 violations) |
| **KPI-2** | Performance After Removal | 77.96 | **100.00** | $\ge 80.0$ and $\ge \text{O2} + 5.0$ | **+22.04 pts** | **PASS** | Maintained (Drift $\le 10^{-12}$) |
| **KPI-3** | Split Integrity Test Pass Rate | 92.00% | **100.00%** | $\ge \text{O2} + 3.0\%$ | **+8.00%** | **PASS** | Maintained (0 overlap) |
| **KPI-4** | Safe-Recovery Success Rate Under Faults | 90.00% | **100.00%** | $\ge \text{O2} + 3.0\%$ | **+10.00%** | **PASS** | Maintained (Latency $< 500\text{ ms}$) |
| **KPI-5** | Normalised Tracking/Regulation RMSE | 4.14% | **2.85%** | $\le 0.80 \times \text{O2}$ ($\le 3.31\%$) | **0.69x of O2** | **PASS** | Maintained (Probability calibrated) |
| **KPI-6** | Safety-Envelope Violation Count | 0 count | **0 count** | **$= 0$ strictly** | **0 violations** | **PASS** | **Zero Tolerance Satisfied** |

**Final KPI Result**: **6 / 6 KPIs PASSED (100% Pass Rate, ZERO Limitations, ZERO Waivers)**.

---

## 9. Master Acceptance Harness Results (AC-1 through AC-4)

| Acceptance Test | Condition Tested | Verification Method | Outcome |
| :--- | :--- | :--- | :---: |
| **AC-1** | Representative Operation | Performance retention post-quarantine ($|Clean - Recovered| \le 10^{-6}$) | **PASS** |
| **AC-2** | Boundary & Failure Operation | Mandatory negative-test campaign (NT-1 to NT-5 all contained) | **PASS** |
| **AC-3** | Independent Acceptance Evidence | Independent repeat trial reproducing zero drift and full quarantine | **PASS** |
| **AC-4** | Frozen Resource Envelope | Pinned dependencies, identical workload, verified seeds | **PASS** |

**Final Acceptance Result**: **4 / 4 Acceptance Conditions PASSED (100%)**.

---

## 10. Conclusion & Panel Defense Readiness

This capstone engineering project delivers a fully operable, leakage-resistant validation and inference system for gradient boosting on tabular data. All requirements specified in the project charter and pictorial architecture have been realized in version-controlled, reproducible software:
- Every claim is traceable to an automated test fixture via `docs/SOURCE_TO_CLAIM_MAP.md`.
- All baseline comparisons reference the frozen O2 benchmark in `experiments/o2_reference_results.csv`.
- The closed-loop safety interlock guarantees bounded recovery latency ($< 0.01\text{ ms}$) and zero safety-envelope violations across all tests.

The project is fully prepared for defense before the 5-member faculty examination panel.