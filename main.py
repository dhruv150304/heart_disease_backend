"""
main.py — FastAPI backend for CardioSensecd /Users/dhruvkansal/Desktop/final_projects/files
uvicorn main:app --reload --host 0.0.0.0 --port 5000

Matches all frontend pages exactly:
  - POST /predict         → Prediction.jsx
  - GET  /dashboard       → Dashboard.jsx
  - GET  /patients        → DoctorDashboard.jsx
  - GET  /patients/{id}   → DoctorDashboard.jsx (single patient)
  - GET  /reports         → Reports.jsx

Run from inside the files/ directory:
    uvicorn main:app --reload --host 0.0.0.0 --port 5000
"""

import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent.parent / "model"
MODEL_PATH    = BASE / "heart_disease_model.pkl"
SCALER_PATH   = BASE / "scaler.pkl"
COLUMNS_PATH  = BASE / "columns.pkl"

model            = None
scaler           = None
expected_columns = None


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
    allow_origins=["*"],
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
    return {"status": "running", "message": "CardioSense API is live 🫀", "docs": "/docs"}


# ── POST /predict ─────────────────────────────────────────────────────────────
# Called by Prediction.jsx
# Returns: { prediction, probability, risk, confidence, label }
@app.post("/predict", tags=["Prediction"])
def predict(data: HeartInput):
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
def get_dashboard():
    return {
        "vitals": [
            {"label": "Heart Rate",    "value": "72",     "unit": "bpm"},
            {"label": "Blood Pressure","value": "118/76", "unit": "mmHg"},
            {"label": "Cholesterol",   "value": "168",    "unit": "mg/dL"},
            {"label": "Blood Sugar",   "value": "92",     "unit": "mg/dL"},
        ],
        "currentRisk":  "Low",
        "riskScore":    18,
        "riskTrend":    [67, 52, 39, 32, 24, 18],
        "predictions": [
            {"date": "26 Apr", "risk": "Low",    "score": 18, "bp": "118/76", "cholesterol": 168},
            {"date": "12 Apr", "risk": "Low",    "score": 24, "bp": "122/78", "cholesterol": 174},
            {"date": "29 Mar", "risk": "Medium", "score": 39, "bp": "130/84", "cholesterol": 196},
            {"date": "15 Mar", "risk": "High",   "score": 67, "bp": "146/92", "cholesterol": 224},
        ],
    }


# ── GET /patients ─────────────────────────────────────────────────────────────
# Called by DoctorDashboard.jsx — returns full patient list
@app.get("/patients", tags=["Doctor"])
def get_patients(risk: str = None, search: str = None):
    patients = [
        {"id": "PT-1024", "name": "Aarav Mehta",     "age": 54, "sex": "Male",   "risk": "High",   "score": 78, "lastReport": "26 Apr 2026", "phone": "+91 98765 12034", "bp": "148/94", "cholesterol": 238, "heartRate": 86, "diagnosis": "Heart disease risk detected",       "notes": "Exercise angina reported. ST slope flat with elevated cholesterol.", "suggestions": ["Schedule ECG review", "Adjust cholesterol management", "Follow up within 7 days"]},
        {"id": "PT-1025", "name": "Isha Kapoor",      "age": 46, "sex": "Female", "risk": "Low",    "score": 18, "lastReport": "25 Apr 2026", "phone": "+91 98111 34490", "bp": "118/76", "cholesterol": 168, "heartRate": 72, "diagnosis": "No heart disease risk detected",    "notes": "Vitals are stable. Continue preventive monitoring.",               "suggestions": ["Routine check-up", "Maintain activity", "Repeat screening in 6 months"]},
        {"id": "PT-1026", "name": "Kabir Singh",      "age": 61, "sex": "Male",   "risk": "Medium", "score": 52, "lastReport": "23 Apr 2026", "phone": "+91 99002 45678", "bp": "136/86", "cholesterol": 211, "heartRate": 80, "diagnosis": "Moderate risk indicators found",     "notes": "Resting BP and cholesterol are above ideal range.",                 "suggestions": ["Lifestyle counselling", "Repeat lipid profile", "Review in 30 days"]},
        {"id": "PT-1027", "name": "Meera Rao",        "age": 39, "sex": "Female", "risk": "Low",    "score": 24, "lastReport": "21 Apr 2026", "phone": "+91 98888 76123", "bp": "122/78", "cholesterol": 174, "heartRate": 76, "diagnosis": "No heart disease risk detected",    "notes": "Slightly elevated stress markers but overall safe range.",          "suggestions": ["Continue monitoring", "Improve sleep routine", "Repeat screening if symptoms appear"]},
        {"id": "PT-1028", "name": "Rohan Malhotra",   "age": 58, "sex": "Male",   "risk": "High",   "score": 84, "lastReport": "20 Apr 2026", "phone": "+91 97654 22310", "bp": "152/96", "cholesterol": 252, "heartRate": 91, "diagnosis": "Heart disease risk detected",       "notes": "High model score with asymptomatic chest pain type and exercise angina.", "suggestions": ["Urgent cardiology appointment", "Monitor BP daily", "Avoid strenuous activity"]},
    ]

    # Optional server-side filtering
    if risk and risk != "All":
        patients = [p for p in patients if p["risk"] == risk]
    if search:
        s = search.lower()
        patients = [p for p in patients if s in p["name"].lower() or s in p["id"].lower() or s in p["phone"]]

    return {"total": len(patients), "patients": patients}


# ── GET /patients/{patient_id} ────────────────────────────────────────────────
# Called by DoctorDashboard when viewing a single patient
@app.get("/patients/{patient_id}", tags=["Doctor"])
def get_patient(patient_id: str):
    all_patients_resp = get_patients()
    match = next((p for p in all_patients_resp["patients"] if p["id"] == patient_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")
    return match


# ── GET /reports ──────────────────────────────────────────────────────────────
# Called by Reports.jsx
@app.get("/reports", tags=["Reports"])
def get_reports():
    return {
        "reports": [
            {"id": 1, "date": "26 Apr 2026", "title": "Monthly Health Summary",    "status": "Complete"},
            {"id": 2, "date": "12 Apr 2026", "title": "Risk Assessment Report",    "status": "Complete"},
            {"id": 3, "date": "29 Mar 2026", "title": "Quarterly Health Report",   "status": "Complete"},
            {"id": 4, "date": "15 Mar 2026", "title": "Annual Health Overview",    "status": "Pending"},
        ]
    }