"""
O4 Split Integrity Verifier: Temporal and Group Separation Verification
Deliverable D4 / CP2 Gate 4 / FR-4

Verifies that validation splits:
  1. Respect chronological ordering (no future leakage: max(t_train) < min(t_test))
  2. Respect entity disjointness (no group contamination: Train_groups ∩ Test_groups = ∅)
  3. Comply with telemetry input/output contracts
"""

from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Set, Dict, Any, Tuple

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "machine_data_v2.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

SPLIT_REPORT_JSON = EXPERIMENTS_DIR / "split_integrity_report.json"


@dataclass
class SplitVerificationReport:
    is_valid: bool
    temporal_valid: bool
    group_valid: bool
    train_rows: int
    test_rows: int
    train_time_range: Tuple[str, str]
    test_time_range: Tuple[str, str]
    time_gap_hours: float
    train_groups: List[str]
    test_groups: List[str]
    overlapping_groups: List[str]
    error_message: Optional[str]
    telemetry_token: str


class SplitIntegrityVerifier:
    """
    Validation engine ensuring strict separation across temporal and group boundaries.
    """

    def __init__(self, time_col: str = "timestamp", group_col: str = "machine_id"):
        self.time_col = time_col
        self.group_col = group_col

    def verify_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        enforce_temporal: bool = True,
        enforce_group: bool = True,
    ) -> SplitVerificationReport:
        train_time = pd.to_datetime(train_df[self.time_col])
        test_time = pd.to_datetime(test_df[self.time_col])

        train_min_t, train_max_t = train_time.min(), train_time.max()
        test_min_t, test_max_t = test_time.min(), test_time.max()

        # 1. Temporal Check: No lookahead leakage
        temporal_valid = True
        time_gap_hours = 0.0
        if enforce_temporal:
            if test_min_t <= train_max_t:
                temporal_valid = False
            else:
                time_gap_hours = float((test_min_t - train_max_t).total_seconds() / 3600.0)

        # 2. Group Check: Mutual exclusion of entities
        train_groups = set(train_df[self.group_col].dropna().unique())
        test_groups = set(test_df[self.group_col].dropna().unique())
        overlapping = train_groups.intersection(test_groups)
        group_valid = (len(overlapping) == 0) if enforce_group else True

        is_valid = temporal_valid and group_valid
        error_msg = None
        if not temporal_valid and not group_valid:
            error_msg = f"Violation: Both Temporal lookahead ({test_min_t} <= {train_max_t}) and Group overlap {list(overlapping)} detected."
        elif not temporal_valid:
            error_msg = f"Temporal Violation: Test set starts at {test_min_t} before training ended at {train_max_t}."
        elif not group_valid:
            error_msg = f"Group Violation: Shared machine IDs found in both train and test: {list(overlapping)}."

        token = f"TOKEN-SPLIT-{'VALID' if is_valid else 'REJECT'}-{abs(hash((str(train_max_t), str(test_min_t), len(train_df)))) % 1000000:06d}"

        report = SplitVerificationReport(
            is_valid=is_valid,
            temporal_valid=temporal_valid,
            group_valid=group_valid,
            train_rows=len(train_df),
            test_rows=len(test_df),
            train_time_range=(str(train_min_t), str(train_max_t)),
            test_time_range=(str(test_min_t), str(test_max_t)),
            time_gap_hours=round(time_gap_hours, 2),
            train_groups=sorted(list(train_groups)),
            test_groups=sorted(list(test_groups)),
            overlapping_groups=sorted(list(overlapping)),
            error_message=error_msg,
            telemetry_token=token,
        )

        return report

    def create_pure_temporal_split(
        self, df: pd.DataFrame, train_ratio: float = 0.80
    ) -> Tuple[pd.DataFrame, pd.DataFrame, SplitVerificationReport]:
        df_sorted = df.sort_values(self.time_col).reset_index(drop=True)
        split_idx = int(len(df_sorted) * train_ratio)
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        report = self.verify_split(train_df, test_df, enforce_temporal=True, enforce_group=False)
        return train_df, test_df, report

    def create_pure_group_split(
        self, df: pd.DataFrame, test_group_count: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, SplitVerificationReport]:
        groups = sorted(df[self.group_col].unique())
        test_groups = set(groups[-test_group_count:])
        train_df = df[~df[self.group_col].isin(test_groups)].copy()
        test_df = df[df[self.group_col].isin(test_groups)].copy()
        report = self.verify_split(train_df, test_df, enforce_temporal=False, enforce_group=True)
        return train_df, test_df, report


def main():
    print("=" * 70)
    print("O4 SPLIT INTEGRITY VERIFIER")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH)
    verifier = SplitIntegrityVerifier()

    # 1. Test Valid Temporal Split
    train_t, test_t, rep_t = verifier.create_pure_temporal_split(df)
    print("\n--- Valid Temporal Split ---")
    print(f"Valid: {rep_t.is_valid} | Token: {rep_t.telemetry_token}")
    print(f"Train: {rep_t.train_time_range[0]} to {rep_t.train_time_range[1]}")
    print(f"Test : {rep_t.test_time_range[0]} to {rep_t.test_time_range[1]}")

    # 2. Test Valid Group Split
    train_g, test_g, rep_g = verifier.create_pure_group_split(df)
    print("\n--- Valid Group Split ---")
    print(f"Valid: {rep_g.is_valid} | Token: {rep_g.telemetry_token}")
    print(f"Train Groups: {rep_g.train_groups}")
    print(f"Test Groups : {rep_g.test_groups}")

    # 3. Test Invalid Split (Negative Test Scenario NT-3)
    train_bad = df.iloc[:3000].copy()
    test_bad = df.iloc[2500:].copy()  # 500 rows overlap!
    rep_bad = verifier.verify_split(train_bad, test_bad, enforce_temporal=True, enforce_group=True)
    print("\n--- Invalid Split (Negative Test NT-3) ---")
    print(f"Valid: {rep_bad.is_valid} (Expected False)")
    print(f"Error: {rep_bad.error_message}")

    summary = {
        "temporal_split_verification": asdict(rep_t),
        "group_split_verification": asdict(rep_g),
        "invalid_split_detection": asdict(rep_bad),
    }

    with open(SPLIT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSplit verification evidence saved to: {SPLIT_REPORT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
