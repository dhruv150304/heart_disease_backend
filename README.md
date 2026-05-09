# HeartGuard API

FastAPI backend for the HeartGuard/CardioSense heart disease prediction app.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Open:

```text
http://localhost:8001/docs
```

## Render Deployment

Use this repository as a Render Web Service. The included `render.yaml` uses:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
PYTHON_VERSION=3.12.3
CORS_ORIGINS=*
```

After the Vercel frontend URL is known, replace `CORS_ORIGINS=*` with that URL, for example:

```text
CORS_ORIGINS=https://your-vercel-app.vercel.app
```

## Model Files

The backend expects these files inside `model/`:

```text
model/heart_disease_model.pkl
model/scaler.pkl
model/columns.pkl
```

They are included so Render can deploy the backend from this repo by itself.

## API

| Method | Route | Description |
| --- | --- | --- |
| GET | `/` | API status |
| GET | `/healthz` | Render health check |
| POST | `/predict` | Heart disease prediction |
| GET | `/dashboard` | Demo dashboard data |
| GET | `/patients` | Demo doctor dashboard data |
| GET | `/patients/{patient_id}` | Single demo patient |
| GET | `/reports` | Demo reports data |

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
