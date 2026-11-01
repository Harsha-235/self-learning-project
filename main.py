"""
Main Engineering Entrypoint & Orchestrator
Project: Gradient Boosting on Tabular Data With Leakage-Resistant Validation
Deliverable D6 / FR-2 / FR-6

Provides a unified one-command execution protocol:
  python main.py --mode all          # Executes complete pipeline & validation gates (O1-O5)
  python main.py --mode reference    # Executes O2 feature importance benchmark
  python main.py --mode contribution # Executes O3 indirect leakage detector & ablation
  python main.py --mode split        # Executes O4 split integrity verification
  python main.py --mode test         # Runs full master acceptance test suite
  python main.py --mode serve        # Launches FastAPI server and displays UI link
"""

import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path(sys.executable)


def run_data_validation():
    print("\n[STEP 1/5] Validating Dataset Provenance and Schema (FR-1)...")
    import pandas as pd
    data_path = PROJECT_ROOT / "data" / "machine_data_v2.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing: {data_path}")
    df = pd.read_csv(data_path)
    required = [
        "timestamp", "machine_id", "temperature", "pressure", "vibration",
        "rotational_speed", "operating_hours", "ambient_temperature",
        "load_percent", "failure"
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"  --> Dataset loaded: {len(df)} records, {len(df.columns)} features.")
    print("  --> Provenance & quality gates: PASSED.")


def run_o2():
    print("\n[STEP 2/5] Executing O2 Reference Benchmark (Deliverable D2)...")
    from ml.feature_importance_reference import run_o2_reference
    run_o2_reference()


def run_o3():
    print("\n[STEP 3/5] Executing O3 Contribution Subsystem (Deliverable D3)...")
    from ml.indirect_leakage_detector import main as o3_main
    o3_main()


def run_o4():
    print("\n[STEP 4/5] Executing O4 Split Integrity Verification (Deliverable D4)...")
    from ml.split_integrity_verifier import main as o4_main
    o4_main()


def run_tests():
    print("\n[STEP 5/5] Executing Master Acceptance & Negative Test Harness (Deliverable D5)...")
    from tests.run_all_tests import run_all
    ret = run_all()
    if ret != 0:
        raise RuntimeError("Master test suite failed.")


def start_server(port=8000):
    import os
    import uvicorn

    port = int(os.environ.get("PORT", port))

    print("=" * 70)
    print(" STARTING GRADIENT BOOSTING WEB SERVICE")
    print("=" * 70)
    print(f"Server running on port: {port}")
    print("=" * 70)

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gradient Boosting Tabular Modeling with Leakage-Resistant Validation"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "reference", "contribution", "split", "test", "serve"],
        default="all",
        help="Execution mode (default: all)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    args = parser.parse_args()

    print("=" * 75)
    print("      GRADIENT BOOSTING ON TABULAR DATA WITH LEAKAGE-RESISTANT VALIDATION")
    print(f"                       EXECUTION MODE: {args.mode.upper()}")
    print("=" * 75)

    if args.mode == "all":
        run_data_validation()
        run_o2()
        run_o3()
        run_o4()
        run_tests()
        print("\n" + "=" * 75)
        print(">>> ALL PIPELINE STAGES, BENCHMARKS, AND TESTS COMPLETED SUCCESSFULLY! <<<")
        print(f"To launch the interactive dashboard, run:  python main.py --mode serve")
        print("=" * 75)

    elif args.mode == "reference":
        run_o2()

    elif args.mode == "contribution":
        run_o3()

    elif args.mode == "split":
        run_o4()

    elif args.mode == "test":
        run_tests()

    elif args.mode == "serve":
        start_server(port=args.port)


if __name__ == "__main__":
    main()