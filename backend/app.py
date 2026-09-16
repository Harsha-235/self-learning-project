from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# GRADIENT BOOSTING LEAKAGE-RESISTANT API
# ==========================================

app = FastAPI(
    title="Gradient Boosting Leakage-Resistant API",
    description="Predict machine failure using the validated Gradient Boosting model.",
    version="1.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "gradient_boosting_model.pkl"
)

model = joblib.load(MODEL_PATH)


# ==========================================
# EXPECTED FEATURES
# ==========================================

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
# INPUT DATA STRUCTURE
# ==========================================

class MachineData(BaseModel):

    temperature: float
    pressure: float
    vibration: float
    rotational_speed: float
    operating_hours: float
    ambient_temperature: float
    load_percent: float


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "project": "Gradient Boosting on Tabular Data With Leakage-Resistant Validation",
        "status": "running",
        "model": "Gradient Boosting",
        "validation": "Leakage-resistant"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: MachineData):

    try:

        # ----------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------

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


        # ----------------------------------
        # CHECK FOR MISSING VALUES
        # ----------------------------------

        if input_data.isnull().any().any():

            raise HTTPException(
                status_code=400,
                detail="Input contains missing values."
            )


        # ----------------------------------
        # CHECK FOR INVALID NUMBERS
        # ----------------------------------

        for column in FEATURES:

            value = input_data[column].iloc[0]

            if not pd.notna(value):

                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {column}."
                )

            if value == float("inf") or value == float("-inf"):

                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {column}."
                )


        # ----------------------------------
        # MODEL PREDICTION
        # ----------------------------------

        prediction = model.predict(
            input_data
        )[0]


        probability = model.predict_proba(
            input_data
        )[0][1]


        # ----------------------------------
        # DETERMINE STATUS
        # ----------------------------------

        if prediction == 1:

            status = "FAILURE_RISK"

        else:

            status = "NORMAL"


        # ----------------------------------
        # RETURN RESULT
        # ----------------------------------

        return {

            "prediction": int(prediction),

            "status": status,

            "failure_probability": round(
                float(probability),
                4
            ),

            "model": "Gradient Boosting",

            "leakage_resistant": True
        }


    # --------------------------------------
    # EXPECTED HTTP ERROR
    # --------------------------------------

    except HTTPException:

        raise


    # --------------------------------------
    # UNEXPECTED ERROR
    # --------------------------------------

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )