# HeartGuard API

FastAPI backend for the HeartGuard AI System, a heart disease prediction application that serves a trained machine learning model through REST API endpoints.

## Live API

Base URL:

```text
https://heartguard-ai-system-backend.onrender.com
```

Health check:

```text
https://heartguard-ai-system-backend.onrender.com/healthz
```

Frontend:

```text
https://heart-guard-ai-system-front.vercel.app
```

## What This Backend Does

- Loads a trained heart disease prediction model at application startup.
- Accepts clinical patient inputs from the React frontend.
- Converts categorical values into the same encoded format used during training.
- Scales the input data before prediction.
- Returns prediction result, disease probability, confidence score, risk level, and readable label.
- Provides supporting demo endpoints for dashboard, doctor view, patients, and reports.

## Tech Stack

| Area | Technology |
| --- | --- |
| API Framework | FastAPI |
| Server | Uvicorn |
| ML Runtime | scikit-learn |
| Data Processing | pandas, NumPy |
| Model Loading | joblib |
| Deployment | Render |

## Project Structure

```text
.
├── main.py
├── model.py
├── schema.py
├── requirements.txt
├── runtime.txt
├── render.yaml
└── model/
    ├── heart_disease_model.pkl
    ├── scaler.pkl
    └── columns.pkl
```

## API Routes

| Method | Route | Description |
| --- | --- | --- |
| GET | `/` | API status |
| GET | `/healthz` | Deployment health check |
| POST | `/predict` | Predict heart disease risk |
| GET | `/dashboard` | Demo dashboard data |
| GET | `/patients` | Demo patient list |
| GET | `/patients/{patient_id}` | Single patient details |
| GET | `/reports` | Demo report list |

## Prediction Request

```json
{
  "Age": 45,
  "Sex": "M",
  "ChestPainType": "ATA",
  "RestingBP": 120,
  "Cholesterol": 200,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 150,
  "ExerciseAngina": "N",
  "Oldpeak": 1.0,
  "ST_Slope": "Up"
}
```

## Prediction Response

```json
{
  "prediction": 0,
  "probability": 22,
  "confidence": 78,
  "risk": "Low",
  "label": "No heart disease risk detected"
}
```

## Local Setup

Clone the repository:

```bash
git clone https://github.com/dhruv150304/HeartGuard-AI-System-backend.git
cd HeartGuard-AI-System-backend
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Open API docs:

```text
http://localhost:8001/docs
```

## Render Deployment

This repository includes `render.yaml`.

Render settings:

```text
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
PYTHON_VERSION=3.12.3
CORS_ORIGINS=https://heart-guard-ai-system-front.vercel.app
```

## Model Files

The following files are required in the `model/` directory:

```text
model/heart_disease_model.pkl
model/scaler.pkl
model/columns.pkl
```

They are included in this backend repository so Render can deploy the API without depending on another repository.

## Important Note

This API provides screening support for educational purposes. It is not a medical diagnosis tool and should not replace professional healthcare advice.

## Author

Dhruv Kansal

GitHub:

```text
https://github.com/dhruv150304
```
