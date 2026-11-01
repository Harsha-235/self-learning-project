
"""
Backend FastAPI Service:
Gradient Boosting with Leakage-Resistant Validation

Deliverable D4 / FR-4, FR-5, FR-6, FR-7, NFR-6

Provides:
- Real-time prediction
- Physical safety-envelope validation
- Safety interlock
- Degraded-mode fallback
- Split integrity verification
- Leakage analysis
- Operational telemetry
"""

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# IMPORT PROJECT MODULES
# ==========================================

from ml.safety_interlock import (
    interlock_controller,
    InterlockState,
    FaultTrigger
)

from ml.split_integrity_verifier import SplitIntegrityVerifier

from ml.indirect_leakage_detector import IndirectLeakageDetector


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Gradient Boosting Leakage-Resistant System API",
    description=(
        "API for machine failure prediction, "
        "split verification and safety interlocks."
    ),
    version="2.1"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# FRONTEND
# ==========================================

FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIR.exists():

    app.mount(
        "/ui",
        StaticFiles(
            directory=str(FRONTEND_DIR),
            html=True
        ),
        name="ui"
    )


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gradient_boosting_model.pkl"
)

try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    print("MODEL LOADING ERROR:", str(e))
    model = None


FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent"
]


# ==========================================
# REQUEST MODEL
# ==========================================

class MachineData(BaseModel):

    temperature: float = Field(...)
    pressure: float = Field(...)
    vibration: float = Field(...)
    rotational_speed: float = Field(...)
    operating_hours: float = Field(...)
    ambient_temperature: float = Field(...)
    load_percent: float = Field(...)


class FaultInjectionRequest(BaseModel):

    trigger: str = Field(...)
    reason: Optional[str] = "Controlled verification scenario"


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "project": (
            "Gradient Boosting on Tabular Data "
            "With Leakage-Resistant Validation"
        ),
        "system_status": "operational",
        "interlock_state": interlock_controller.state.value,
        "model": "Gradient Boosting Classifier",
        "validation_strategy": (
            "Temporal & Group Disjoint + "
            "Indirect Leakage Quarantine"
        ),
        "dashboard_ui": "/ui/",
        "api_version": "2.1"
    }


# ==========================================
# HEALTH
# ==========================================

@app.get("/health")
def health():

    return {
        "status": (
            "healthy"
            if interlock_controller.state
            not in [
                InterlockState.INTERLOCK_LATCHED,
                InterlockState.DEGRADED_MODE
            ]
            else "interlock_latched"
        ),
        "model_loaded": model is not None,
        "interlock_state": interlock_controller.state.value,
        "timestamp": time.time()
    }


# ==========================================
# SYSTEM STATUS
# ==========================================

@app.get("/system/status")
def get_system_status():

    return {
        "interlock_state": interlock_controller.state.value,

        "active_trigger": (
            interlock_controller.active_trigger.value
            if interlock_controller.active_trigger
            else None
        ),

        "safety_envelope_limits": (
            interlock_controller.safety_envelope_limits
        ),

        "recent_recovery_times_ms": (
            interlock_controller.recovery_times_ms[-5:]
        ),

        "events_count": len(
            interlock_controller.event_history
        ),

        "model_loaded": model is not None,

        "is_degraded": interlock_controller.state in [
            InterlockState.INTERLOCK_LATCHED,
            InterlockState.DEGRADED_MODE
        ]
    }


# ==========================================
# MACHINE FAILURE PREDICTION
# ==========================================

@app.post("/predict")
def predict(data: MachineData):

    sensor_dict = data.model_dump()

    # STEP 1: MODEL AVAILABILITY CHECK

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Gradient Boosting model is not loaded."
        )

    # STEP 2: SAFETY ENVELOPE VALIDATION

    within_envelope, envelope_violation = (
        interlock_controller.check_sensor_envelope(
            sensor_dict
        )
    )

    if not within_envelope:

        interlock_controller.trip_interlock(
            FaultTrigger.SENSOR_ENVELOPE_TRIP,
            preconditions=sensor_dict,
            observed_metrics={
                "violation": envelope_violation
            }
        )

        raise HTTPException(
            status_code=400,
            detail={
                "status": "REJECTED",
                "safety_check": "FAILED",
                "system_state": (
                    interlock_controller.state.value
                ),
                "message": envelope_violation
            }
        )

    # STEP 3: CHECK INTERLOCK STATE

    if interlock_controller.state in [
        InterlockState.INTERLOCK_LATCHED,
        InterlockState.DEGRADED_MODE
    ]:

        return (
            interlock_controller
            .get_conservative_fallback_prediction(
                sensor_dict
            )
        )

    # STEP 4: GENERATE MODEL INPUT

    try:

        input_data = pd.DataFrame(
            [[
                data.temperature,
                data.pressure,
                data.vibration,
                data.rotational_speed,
                data.operating_hours,
                data.ambient_temperature,
                data.load_percent
            ]],
            columns=FEATURES
        )

        # STEP 5: PREDICTION

        prediction = int(
            model.predict(input_data)[0]
        )

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        status = (
            "FAILURE_RISK"
            if prediction == 1
            else "NORMAL"
        )

        return {
            "prediction": prediction,
            "status": status,
            "failure_probability": round(
                probability,
                4
            ),
            "model": (
                "Gradient Boosting (Leakage-Resistant)"
            ),
            "interlock_state": (
                interlock_controller.state.value
            ),
            "is_degraded": False,
            "safety_check": "PASSED",
            "system_state": "HEALTHY"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ==========================================
# MANUAL / CONTROLLED INTERLOCK TRIP
# ==========================================

@app.post("/interlock/trip")
def trip_interlock(request: FaultInjectionRequest):

    trigger_map = {

        "NT-1": FaultTrigger.NT1_DEPLOYMENT_COLLAPSE,

        "NT-2": FaultTrigger.NT2_HIDDEN_LEAKAGE,

        "NT-3": FaultTrigger.NT3_SPLIT_VIOLATION,

        "NT-4": FaultTrigger.NT4_SIM_TO_REAL_MISMATCH,

        "NT-5": FaultTrigger.NT5_WORST_CASE_RISK,

        "MANUAL": FaultTrigger.MANUAL_INSPECTION_HOLD
    }

    trigger = trigger_map.get(
        request.trigger.upper(),
        FaultTrigger.MANUAL_INSPECTION_HOLD
    )

    evt = interlock_controller.trip_interlock(
        trigger=trigger,
        preconditions={
            "requested_by": "operator_or_test_fixture",
            "reason": request.reason
        }
    )

    return {
        "status": "INTERLOCK_LATCHED",
        "event_id": evt.event_id,
        "trigger": evt.trigger,
        "safe_action": evt.safe_action_taken,
        "message": (
            "Model predictions quarantined. "
            "Degraded safe fallback active."
        )
    }


# ==========================================
# INTERLOCK RECOVERY
# ==========================================

@app.post("/interlock/recover")
def recover_interlock():

    success, rec_time, evt = (
        interlock_controller.execute_recovery(
            verification_routine_passed=True
        )
    )

    return {
        "status": (
            "RECOVERED"
            if success
            else "INTERLOCK_HOLD"
        ),
        "recovery_time_ms": rec_time,
        "event_id": evt.event_id,
        "message": (
            "Recovery completed."
            if success
            else "Recovery failed."
        )
    }


# ==========================================
# OPERATIONAL TELEMETRY
# ==========================================

@app.get("/telemetry")
def get_telemetry():

    recovery_times = (
        interlock_controller.recovery_times_ms
    )

    return {
        "events": [
            asdict(e)
            for e in interlock_controller.event_history[-10:]
        ],

        "total_events": len(
            interlock_controller.event_history
        ),

        "mean_recovery_time_ms": (
            round(float(np.mean(recovery_times)), 3)
            if recovery_times
            else None
        )
    }


# ==========================================
# SPLIT INTEGRITY VERIFICATION
# ==========================================

@app.post("/validation/verify-split")
def verify_split_endpoint():

    data_path = (
        PROJECT_ROOT
        / "data"
        / "machine_data_v2.csv"
    )

    if not data_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    df = pd.read_csv(data_path)

    verifier = SplitIntegrityVerifier()

    train_t, test_t, rep_t = (
        verifier.create_pure_temporal_split(df)
    )

    train_g, test_g, rep_g = (
        verifier.create_pure_group_split(df)
    )

    return {
        "temporal_split": {
            "valid": rep_t.is_valid,
            "train_range": rep_t.train_time_range,
            "test_range": rep_t.test_time_range,
            "token": rep_t.telemetry_token
        },

        "group_split": {
            "valid": rep_g.is_valid,
            "train_groups": rep_g.train_groups,
            "test_groups": rep_g.test_groups,
            "token": rep_g.telemetry_token
        }
    }


# ==========================================
# INDIRECT LEAKAGE ANALYSIS
# ==========================================

@app.post("/leakage/analyze")
def analyze_leakage_endpoint():

    data_path = (
        PROJECT_ROOT
        / "data"
        / "machine_data_v2.csv"
    )

    if not data_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    df = pd.read_csv(data_path)

    detector = IndirectLeakageDetector()

    results = detector.fit_detect(
        df[FEATURES],
        df["failure"]
    )

    return {
        "features_analyzed": [

            {
                "feature": r.feature,
                "mutual_info": r.mutual_info,
                "proxy_auc": r.reconstruction_auc,
                "shadow_ratio": r.shadow_ratio,
                "rubric_score": r.rubric_score,
                "verdict": r.verdict
            }

            for r in results
        ]
    }
