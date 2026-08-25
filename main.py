"""
FastAPI backend for the CardioSense app.

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8001

Deploy on Render with:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import joblib
import os
import pandas as pd
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal
from auth import current_user, get_supabase, require_clinician

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent / "model"
MODEL_PATH    = BASE / "heart_disease_model.pkl"
SCALER_PATH   = BASE / "scaler.pkl"
COLUMNS_PATH  = BASE / "columns.pkl"

model            = None
scaler           = None
expected_columns = None


def get_allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173").strip()
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, expected_columns
    try:
        model            = joblib.load(MODEL_PATH)
        scaler           = joblib.load(SCALER_PATH)
        expected_columns = list(joblib.load(COLUMNS_PATH))
        print(f"✅ Model loaded | Columns: {expected_columns}")
    except FileNotFoundError as e:
        raise RuntimeError(f"❌ Could not load model files: {e}")
    yield
    model = scaler = expected_columns = None


app = FastAPI(
    title="CardioSense API",
    description="FastAPI backend for the CardioSense heart disease prediction app.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class HeartInput(BaseModel):
    Age: int
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ATA", "NAP", "TA", "ASY"]
    RestingBP: int
    Cholesterol: int
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float
    ST_Slope: Literal["Up", "Flat", "Down"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 45, "Sex": "M", "ChestPainType": "ATA",
                "RestingBP": 120, "Cholesterol": 200, "FastingBS": 0,
                "RestingECG": "Normal", "MaxHR": 150,
                "ExerciseAngina": "N", "Oldpeak": 1.0, "ST_Slope": "Up"
            }
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_risk_label(probability: int) -> str:
    if probability >= 70:
        return "High"
    elif probability >= 40:
        return "Medium"
    return "Low"


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
def root():
    return {"status": "running", "message": "CardioSense API is live", "docs": "/docs"}


@app.get("/healthz", tags=["General"])
def healthcheck():
    return {"status": "ok"}


# ── POST /predict ─────────────────────────────────────────────────────────────
# Called by Prediction.jsx
# Returns: { prediction, probability, risk, confidence, label }
@app.post("/predict", tags=["Prediction"])
def predict(data: HeartInput, user: dict = Depends(current_user)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Build numeric base
    input_data = pd.DataFrame([{
        "Age":         data.Age,
        "RestingBP":   data.RestingBP,
        "Cholesterol": data.Cholesterol,
        "FastingBS":   data.FastingBS,
        "MaxHR":       data.MaxHR,
        "Oldpeak":     data.Oldpeak,
    }])

    # One-hot encode exactly as training did
    input_data["Sex_M"]              = 1 if data.Sex == "M" else 0
    input_data["ChestPainType_ATA"]  = 1 if data.ChestPainType == "ATA" else 0
    input_data["ChestPainType_NAP"]  = 1 if data.ChestPainType == "NAP" else 0
    input_data["ChestPainType_TA"]   = 1 if data.ChestPainType == "TA"  else 0
    input_data["RestingECG_Normal"]  = 1 if data.RestingECG == "Normal" else 0
    input_data["RestingECG_ST"]      = 1 if data.RestingECG == "ST"     else 0
    input_data["ExerciseAngina_Y"]   = 1 if data.ExerciseAngina == "Y"  else 0
    input_data["ST_Slope_Flat"]      = 1 if data.ST_Slope == "Flat"     else 0
    input_data["ST_Slope_Up"]        = 1 if data.ST_Slope == "Up"       else 0

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_columns]

    scaled       = scaler.transform(input_data)
    prediction   = int(model.predict(scaled)[0])

    # Probability: use predict_proba if available, else derive from prediction
    try:
        proba       = model.predict_proba(scaled)[0]
        probability = int(round(proba[1] * 100))   # disease probability %
    except AttributeError:
        probability = 76 if prediction == 1 else 22

    confidence = max(probability, 100 - probability)
    risk       = compute_risk_label(probability)

    try:
        get_supabase().table("predictions").insert({
            "patient_id": user["id"],
            "input": data.model_dump(),
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "risk": risk,
            "model_version": "heart-disease-model-v1",
        }).execute()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Prediction could not be saved securely.") from exc

    return {
        "prediction":  prediction,
        "probability": probability,
        "confidence":  confidence,
        "risk":        risk,
        "label":       "Heart disease risk detected" if prediction == 1 else "No heart disease risk detected",
    }


# ── GET /dashboard ────────────────────────────────────────────────────────────
# Called by Dashboard.jsx — returns vitals, prediction history, risk trend
@app.get("/dashboard", tags=["Dashboard"])
def get_dashboard(user: dict = Depends(current_user)):
    response = get_supabase().table("predictions").select("id, probability, confidence, risk, prediction, created_at, input").eq("patient_id", user["id"]).order("created_at", desc=True).limit(20).execute()
    predictions = response.data or []
    latest = predictions[0] if predictions else None
    return {"currentRisk": latest["risk"] if latest else None, "riskScore": latest["probability"] if latest else None, "predictions": predictions}


# ── GET /patients ─────────────────────────────────────────────────────────────
# Called by DoctorDashboard.jsx — returns full patient list
@app.get("/patients", tags=["Doctor"])
def get_patients(risk: str = None, search: str = None, clinician: dict = Depends(require_clinician)):
    db = get_supabase()
    assignments = db.table("clinician_patients").select("patient_id").eq("clinician_id", clinician["id"]).execute().data or []
    patients = []
    for assignment in assignments:
        patient_id = assignment["patient_id"]
        profile = db.table("profiles").select("id, full_name").eq("id", patient_id).single().execute().data
        latest = db.table("predictions").select("id, risk, probability, confidence, prediction, created_at").eq("patient_id", patient_id).order("created_at", desc=True).limit(1).execute().data
        if profile and latest:
            record = {"id": profile["id"], "name": profile["full_name"], **latest[0]}
            if (not risk or risk == "All" or record["risk"] == risk) and (not search or search.lower() in record["name"].lower()):
                patients.append(record)
    return {"total": len(patients), "patients": patients}


# ── GET /patients/{patient_id} ────────────────────────────────────────────────
# Called by DoctorDashboard when viewing a single patient
@app.get("/patients/{patient_id}", tags=["Doctor"])
def get_patient(patient_id: str, clinician: dict = Depends(require_clinician)):
    all_patients_resp = get_patients(clinician=clinician)
    match = next((p for p in all_patients_resp["patients"] if p["id"] == patient_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")
    return match


# ── GET /reports ──────────────────────────────────────────────────────────────
# Called by Reports.jsx
@app.get("/reports", tags=["Reports"])
def get_reports(user: dict = Depends(current_user)):
    reports = get_supabase().table("predictions").select("id, risk, probability, confidence, created_at").eq("patient_id", user["id"]).order("created_at", desc=True).execute().data or []
    return {"reports": reports}
