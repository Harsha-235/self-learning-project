# Deliverable D1: Architecture Decision Records (ADRs)

---

### ADR-001: Selection of Gradient Boosted Decision Trees (GBDT)
* **Status**: Accepted
* **Context**: The project domain requires high-performance classification of tabular sensor records for condition-based maintenance.
* **Decision**: We adopt Scikit-learn's `GradientBoostingClassifier` with frozen hyperparameters (`n_estimators=100`, `learning_rate=0.05`, `max_depth=3`, `random_state=42`).
* **Consequences**: Tree ensembles capture non-linear sensor interactions and operating regimes without intensive feature scaling. However, GBDTs readily overfit to indirect leakage or proxy target representations, requiring an explicit leakage detection and validation verification harness.

---

### ADR-002: Dual-Method Feature Importance Reference for O2 Baseline
* **Status**: Accepted
* **Context**: Gate 2 (O2) mandates comparing at least two feasible reference approaches for feature importance and target relationship analysis.
* **Decision**: We benchmark:
  1. **Method A (MDI - Mean Decrease in Impurity)**: Fast, built-in Gini gain calculation across split nodes.
  2. **Method B (Permutation Importance)**: Out-of-fold feature shuffling to observe drop in model scoring metric.
* **Trade-off Analysis**: MDI is computationally trivial ($O(N)$ with tree evaluation) but notoriously biased towards continuous features and fails to discern whether high importance is legitimate or from target leakage. Permutation Importance is computationally heavier ($O(K \times N)$ where $K$ is number of features) but directly reflects true predictive degradation and does not falsely inflate collinear proxy features.
* **Selected Reference**: Permutation Importance is selected as the primary reference against which the O3 contribution is evaluated, with MDI retained as baseline comparison.

---

### ADR-003: Multi-Signal Indirect Target-Encoding Detection Strategy (O3)
* **Status**: Accepted
* **Context**: O3 requires detecting features that encode the target indirectly (e.g. proxy sensors, post-failure maintenance records, noisy target derivatives).
* **Decision**: We implement an ensemble detection strategy in `ml/indirect_leakage_detector.py` combining:
  1. **Nonlinear Mutual Information (MI)**: Quantifies information sharing between feature and target independent of monotonic linearity.
  2. **Univariate Target Reconstruction AUC**: Measures whether a low-capacity classifier trained only on that feature achieves anomalous separation ($AUC \ge 0.85$).
  3. **Out-of-Fold Shift Delta**: Assesses whether training vs out-of-fold error diverges sharply.
  4. **Shadow-Feature Sensitivity**: Adds permuted shadow noise columns to test if the candidate feature suppresses all physical sensors.
* **Consequences**: Enables pre-deployment detection of subtle target leakage with a tunable confidence rubric ($0–100$).

---

### ADR-004: Temporal-Group Disjoint Split Validation Architecture (O4)
* **Status**: Accepted
* **Context**: Validation splits in industrial IoT frequently suffer from temporal leakage (future data in training) or group leakage (same physical asset in train and test).
* **Decision**: We design `ml/split_integrity_verifier.py` with strict pre-validation checks:
  1. **Temporal Disjointness**: Verifies $\max(t_{\text{train}}) < \min(t_{\text{test}})$.
  2. **Group Disjointness**: Verifies $\text{Entities}_{\text{train}} \cap \text{Entities}_{\text{test}} = \emptyset$.
  3. Rejection and automatic re-splitting if contamination is detected.
* **Consequences**: Prevents deceptive cross-contamination and ensures deployment fidelity.

---

### ADR-005: Safety Interlock State Machine & Degraded-Mode Recovery
* **Status**: Accepted
* **Context**: Problem statement mandates that upon encountering failure conditions (NT-1 through NT-5), the system must invoke a safe interlock, expose the fault, enter degraded mode, and demonstrate bounded recovery.
* **Decision**: Implement `ml/safety_interlock.py` containing an asynchronous state machine with states: `HEALTHY`, `INTERLOCK_LATCHED`, `DEGRADED_MODE`, and `RECOVERED`. The controller rejects unverified predictions, serves safe conservative fallback estimates, logs recovery latency ($<500\text{ ms}$), and stores structured audit trails.
