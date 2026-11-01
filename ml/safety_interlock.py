
"""
O4 Safety Interlock Controller & Degraded-Mode Recovery Subsystem
Deliverable D4 / FR-7 / NFR-6 / NT-1 to NT-5

Implements:
    HEALTHY
    INTERLOCK_LATCHED
    DEGRADED_MODE
    RECOVERED

Validates all seven sensor readings before ML prediction.
"""

import time
import json
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

INTERLOCK_LOG_FILE = EXPERIMENTS_DIR / "safety_interlock_log.json"


class InterlockState(str, Enum):
    HEALTHY = "HEALTHY"
    INTERLOCK_LATCHED = "INTERLOCK_LATCHED"
    DEGRADED_MODE = "DEGRADED_MODE"
    RECOVERED = "RECOVERED"


class FaultTrigger(str, Enum):
    NT1_DEPLOYMENT_COLLAPSE = "NT-1: Validation Performance Collapse in Deployment"
    NT2_HIDDEN_LEAKAGE = "NT-2: Invisible Indirect Leakage Exploitation"
    NT3_SPLIT_VIOLATION = "NT-3: Validation Split Structure Contamination"
    NT4_SIM_TO_REAL_MISMATCH = "NT-4: Simulation-to-Deployment Transfer Failure"
    NT5_WORST_CASE_RISK = "NT-5: Optimization Worsened Worst-Case Risk"
    MANUAL_INSPECTION_HOLD = "MANUAL: Operator Manual Hold"
    SENSOR_ENVELOPE_TRIP = "SAFETY: Sensor Operating Envelope Trip"


@dataclass
class InterlockEvent:
    event_id: str
    timestamp: float
    state_before: str
    state_after: str
    trigger: str
    preconditions: Dict[str, Any]
    observed_metrics: Dict[str, Any]
    safe_action_taken: str
    recovery_time_ms: Optional[float]
    residual_risk_decision: str


class SafetyInterlockController:
    """
    Safety controller that validates sensor readings
    and blocks unsafe ML predictions.
    """

    def __init__(self):

        self.state = InterlockState.HEALTHY
        self.active_trigger: Optional[FaultTrigger] = None

        self.event_history: List[InterlockEvent] = []
        self.recovery_times_ms: List[float] = []

        # Proposed operating limits.
        # Confirm these against the machine's actual specifications.

        self.safety_envelope_limits = {
            "temperature_min": 0.0,
            "temperature_max": 110.0,

            "pressure_min": 0.0,
            "pressure_max": 130.0,

            "vibration_min": 0.0,
            "vibration_max": 5.0,

            "speed_min": 800.0,
            "speed_max": 2200.0,

            "operating_hours_min": 0.0,

            "ambient_temperature_min": -50.0,
            "ambient_temperature_max": 80.0,

            "load_min": 0.0,
            "load_max": 100.0
        }

        self.load_history()

    def load_history(self):

        if INTERLOCK_LOG_FILE.exists():

            try:
                with open(
                    INTERLOCK_LOG_FILE,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                    self.event_history = [
                        InterlockEvent(**item)
                        for item in data
                    ]

            except Exception:
                self.event_history = []

    def save_history(self):

        with open(
            INTERLOCK_LOG_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                [asdict(e) for e in self.event_history],
                f,
                indent=4
            )

    # ==========================================
    # SENSOR SAFETY VALIDATION
    # ==========================================

    def check_sensor_envelope(
        self,
        sensor_data: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:

        """
        Validate all seven sensor readings.

        Returns:
            (True, None) if all readings are valid.
            (False, error_message) if any reading is invalid.
        """

        required_sensors = [
            "temperature",
            "pressure",
            "vibration",
            "rotational_speed",
            "operating_hours",
            "ambient_temperature",
            "load_percent"
        ]

        # Check that all seven readings exist
        for sensor in required_sensors:

            if sensor not in sensor_data:
                return False, f"Missing sensor reading: {sensor}"

            value = sensor_data[sensor]

            # Reject None, NaN, infinity and non-numeric values
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                return False, f"Invalid sensor reading: {sensor}"

            if not np.isfinite(value):
                return False, f"Non-finite sensor reading: {sensor}"

        # Read validated sensor values
        temp = float(sensor_data["temperature"])
        press = float(sensor_data["pressure"])
        vib = float(sensor_data["vibration"])
        speed = float(sensor_data["rotational_speed"])
        hours = float(sensor_data["operating_hours"])
        ambient = float(sensor_data["ambient_temperature"])
        load = float(sensor_data["load_percent"])

        limits = self.safety_envelope_limits

        # Machine temperature
        if not (
            limits["temperature_min"]
            <= temp
            <= limits["temperature_max"]
        ):
            return False, (
                f"Temperature {temp}°C is outside the permitted range."
            )

        # Pressure
        if not (
            limits["pressure_min"]
            <= press
            <= limits["pressure_max"]
        ):
            return False, (
                f"Pressure {press} kPa is outside the permitted range."
            )

        # Vibration
        if not (
            limits["vibration_min"]
            <= vib
            <= limits["vibration_max"]
        ):
            return False, (
                f"Vibration {vib} mm/s is outside the permitted range."
            )

        # Rotational speed
        if not (
            limits["speed_min"]
            <= speed
            <= limits["speed_max"]
        ):
            return False, (
                f"Rotational speed {speed} RPM is outside the permitted range."
            )

        # Operating hours
        if hours < limits["operating_hours_min"]:
            return False, "Operating hours cannot be negative."

        # Ambient temperature
        if not (
            limits["ambient_temperature_min"]
            <= ambient
            <= limits["ambient_temperature_max"]
        ):
            return False, (
                f"Ambient temperature {ambient}°C is outside "
                f"the permitted range "
                f"({limits['ambient_temperature_min']} to "
                f"{limits['ambient_temperature_max']}°C)."
            )

        # Machine load
        if not (
            limits["load_min"]
            <= load
            <= limits["load_max"]
        ):
            return False, (
                f"Machine load {load}% is outside the permitted range."
            )

        return True, None

    # ==========================================
    # INTERLOCK TRIP
    # ==========================================

    def trip_interlock(
        self,
        trigger: FaultTrigger,
        preconditions: Optional[Dict[str, Any]] = None,
        observed_metrics: Optional[Dict[str, Any]] = None
    ) -> InterlockEvent:

        t0 = time.time()

        prev_state = self.state.value

        self.state = InterlockState.INTERLOCK_LATCHED
        self.active_trigger = trigger

        event = InterlockEvent(
            event_id=f"EVT-TRIP-{int(t0 * 1000) % 1000000:06d}",
            timestamp=t0,
            state_before=prev_state,
            state_after=self.state.value,
            trigger=trigger.value,
            preconditions=preconditions or {},
            observed_metrics=observed_metrics or {},
            safe_action_taken=(
                "LATCH_INTERLOCK: Block ML prediction and "
                "activate conservative degraded heuristic."
            ),
            recovery_time_ms=None,
            residual_risk_decision=(
                "HOLD_DEPLOYMENT: Unsafe input quarantined."
            )
        )

        self.event_history.append(event)
        self.save_history()

        return event

    # ==========================================
    # DEGRADED MODE
    # ==========================================

    def enter_degraded_mode(self):

        if self.state == InterlockState.INTERLOCK_LATCHED:
            self.state = InterlockState.DEGRADED_MODE

    def get_conservative_fallback_prediction(
        self,
        sensor_data: Dict[str, float]
    ) -> Dict[str, Any]:

        """
        Deterministic fallback.
        Does not use the ML model.
        """

        vib = sensor_data.get("vibration", 1.0)
        temp = sensor_data.get("temperature", 60.0)

        is_risky = (
            vib > 2.5
            or temp > 80.0
            or (vib > 2.0 and temp > 75.0)
        )

        prob = 0.85 if is_risky else 0.15

        return {
            "prediction": 1 if is_risky else 0,
            "status": (
                "FAILURE_RISK (DEGRADED_FALLBACK)"
                if is_risky
                else "NORMAL (DEGRADED_FALLBACK)"
            ),
            "failure_probability": prob,
            "model": "Deterministic Conservative Physical Heuristic",
            "interlock_state": self.state.value,
            "is_degraded": True,
            "warning": (
                "Safety interlock active. "
                "Serving fail-safe deterministic response."
            )
        }

    # ==========================================
    # RECOVERY
    # ==========================================

    def execute_recovery(
        self,
        verification_routine_passed: bool = True
    ) -> Tuple[bool, float, InterlockEvent]:

        t_start = time.perf_counter()
        t_wall = time.time()

        prev_state = self.state.value

        if self.state not in [
            InterlockState.INTERLOCK_LATCHED,
            InterlockState.DEGRADED_MODE
        ]:
            event = InterlockEvent(
                event_id=f"EVT-REC-NOOP-{int(t_wall*1000)%1000000:06d}",
                timestamp=t_wall,
                state_before=prev_state,
                state_after=prev_state,
                trigger="RECOVERY_NOT_REQUIRED",
                preconditions={},
                observed_metrics={},
                safe_action_taken="NO_ACTION: No active interlock.",
                recovery_time_ms=0.0,
                residual_risk_decision="NO_ACTIVE_FAULT"
            )

            return True, 0.0, event

        if not verification_routine_passed:

            event = InterlockEvent(
                event_id=f"EVT-REC-FAIL-{int(t_wall*1000)%1000000:06d}",
                timestamp=t_wall,
                state_before=prev_state,
                state_after=prev_state,
                trigger="RECOVERY_ATTEMPT",
                preconditions={"interlock_active": True},
                observed_metrics={"verification_passed": False},
                safe_action_taken=(
                    "RETAIN_HOLD: Verification failed."
                ),
                recovery_time_ms=None,
                residual_risk_decision="UNRESOLVED_HAZARD"
            )

            self.event_history.append(event)
            self.save_history()

            return False, 0.0, event

        self.state = InterlockState.RECOVERED
        self.active_trigger = None

        recovery_time_ms = round(
            (time.perf_counter() - t_start) * 1000.0,
            3
        )

        self.recovery_times_ms.append(recovery_time_ms)

        event = InterlockEvent(
            event_id=f"EVT-REC-OK-{int(t_wall*1000)%1000000:06d}",
            timestamp=t_wall,
            state_before=prev_state,
            state_after=self.state.value,
            trigger="VERIFIED_SAFE_RECOVERY",
            preconditions={
                "verification_routine_passed": True
            },
            observed_metrics={
                "recovery_time_ms": recovery_time_ms
            },
            safe_action_taken="RESTORE_AUTHORITY",
            recovery_time_ms=recovery_time_ms,
            residual_risk_decision="RECOVERY_ACCEPTED"
        )

        self.event_history.append(event)
        self.save_history()

        return True, recovery_time_ms, event


# Global singleton controller instance
interlock_controller = SafetyInterlockController()


if __name__ == "__main__":

    print("=" * 65)
    print("SAFETY INTERLOCK VALIDATION TEST")
    print("=" * 65)

    ctrl = SafetyInterlockController()

    test_data = {
        "temperature": 50,
        "pressure": 100,
        "vibration": 0.5,
        "rotational_speed": 1500,
        "operating_hours": 50,
        "ambient_temperature": 25000000,
        "load_percent": 40
    }

    valid, message = ctrl.check_sensor_envelope(test_data)

    print("Sensor validation:", valid)
    print("Message:", message)

    if not valid:
        ctrl.trip_interlock(
            FaultTrigger.SENSOR_ENVELOPE_TRIP,
            preconditions=test_data,
            observed_metrics={"violation": message}
        )

    print("Final state:", ctrl.state.value)
    print("=" * 65)
