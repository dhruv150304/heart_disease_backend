"""
model.py — Train the Heart Disease model and save it as heart_model.pkl
Run this script once before starting the FastAPI server:
    python model.py
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── UCI Cleveland Heart Disease dataset (303 samples, 13 features) ──────────
# You can replace this with: pd.read_csv("your_dataset.csv")
from sklearn.datasets import fetch_openml

def train_and_save_model(save_path: str = "heart_model.pkl"):
    print("📦 Loading dataset...")

    # Load Cleveland Heart Disease dataset from OpenML
    dataset = fetch_openml(name="heart-c", version=1, as_frame=True, parser="auto")
    df = dataset.frame.copy()

    # Target: 'class' column → 0 (no disease) or 1 (disease)
    df["target"] = df["class"].apply(lambda x: 0 if str(x).strip() == "0" else 1)
    df = df.drop(columns=["class"])

    # Features
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    # Coerce to numeric, drop rows with missing values
    for col in feature_names:
        df[col] = df[col].apply(lambda x: float(str(x).strip()) if str(x).strip() not in ["?", "nan", ""] else np.nan)
    df = df.dropna(subset=feature_names + ["target"])

    X = df[feature_names].values.astype(float)
    y = df["target"].values.astype(int)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Pipeline: scaler + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    print("🏋️  Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"📊 F1 Score  : {f1:.4f}")
    print(f"📊 Accuracy  : {acc:.4f}")

    # Save model + metadata
    model_bundle = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "model_name": "RandomForestClassifier",
    }

    with open(save_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"💾 Model saved to → {save_path}")
    return model_bundle


if __name__ == "__main__":
    train_and_save_model()
