"""
Master Verification and Acceptance Runner
Executes the full test suite for Capstone Panel Review:
  1. O2 Reference Oracle (MDI vs Permutation)
  2. O3 Contribution Oracle (Indirect Leakage Detector)
  3. O4 Split Integrity & NT-1..NT-5 Negative Test Campaign
  4. AC-1 to AC-4 Unified Acceptance Harness
  5. KPI-1 to KPI-6 Target Verification
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path(sys.executable)

TEST_SUITE = [
    ("O2 Reference Oracle", "tests/test_o2_reference.py"),
    ("O3 Contribution Oracle", "tests/test_o3_contribution.py"),
    ("O4/O5 Mandatory Negative Tests (NT-1 to NT-5)", "tests/negative_tests.py"),
    ("AC-1 to AC-4 Acceptance Harness", "tests/acceptance_tests.py"),
    ("KPI-1 to KPI-6 Verification Engine", "tests/kpi_tests.py"),
]


def run_all():
    print("=" * 75)
    print("      GRADIENT BOOSTING WITH LEAKAGE-RESISTANT VALIDATION")
    print("             MASTER PANEL ACCEPTANCE TEST SUITE")
    print("=" * 75)

    all_passed = True
    summary = []

    for name, script_rel in TEST_SUITE:
        script_path = PROJECT_ROOT / script_rel
        print(f"\n[RUNNING] {name} ({script_rel})...")
        res = subprocess.run([str(PYTHON_EXE), str(script_path)], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  --> [PASS] {name}")
            summary.append((name, "PASS"))
        else:
            print(f"  --> [FAIL] {name}")
            print(res.stderr)
            summary.append((name, "FAIL"))
            all_passed = False

    print("\n" + "=" * 75)
    print("                    MASTER TEST SUMMARY")
    print("=" * 75)
    for name, status in summary:
        print(f"  {status:6s} | {name}")
    print("=" * 75)

    if all_passed:
        print("\n>>> ALL TESTS PASSED! FULL SYSTEM CONTRACT MET FOR PANEL DEFENSE! <<<\n")
        return 0
    else:
        print("\n>>> ONE OR MORE TESTS FAILED! <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
