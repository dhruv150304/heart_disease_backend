# 🫀 Heart Disease Prediction — FastAPI Backend

ML-powered REST API using **Random Forest** + **F1 Score** evaluation on the UCI Cleveland Heart Disease dataset.

---

## 📁 Project Structure

```
heart_disease_api/
├── main.py          # FastAPI app with all routes
├── model.py         # Model training & saving script
├── schema.py        # Pydantic request/response schemas
├── requirements.txt # Python dependencies
└── heart_model.pkl  # Auto-generated after running model.py
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train & save the model
```bash
python model.py
```
This downloads the Cleveland dataset, trains a Random Forest, prints the **F1 Score**, and saves `heart_model.pkl`.

### 3. Start the API server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open Swagger UI
```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Health check |
| GET | `/model-info` | Model metadata + F1 score |
| POST | `/predict` | Single patient prediction |
| POST | `/predict-batch` | Batch prediction (max 100) |

---

## 🧪 Sample Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
    "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
  }'
```

### Sample Response
```json
{
  "prediction": 1,
  "label": "Heart Disease Detected 🚨",
  "probability_no_disease": 0.12,
  "probability_disease": 0.88,
  "f1_score_on_test": 0.8421
}
```

---

## 🔁 Using Your Own Model

Replace the content in `model.py` with your own training code. Make sure to save the same bundle format:
```python
model_bundle = {
    "pipeline": your_pipeline,       # sklearn pipeline or model
    "feature_names": [...],          # list of feature names in order
    "f1_score": f1,
    "accuracy": acc,
    "model_name": "YourModelName",
}
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
```

---

## 📊 Input Features (Cleveland UCI)

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | 0 = Female, 1 = Male |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0–2) |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Major vessels colored by fluoroscopy (0–3) |
| thal | Thalassemia type (0–3) |
