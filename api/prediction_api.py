from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("../models/best_xgboost_model.pkl")
features = joblib.load("../models/model_features.pkl")

@app.post("/predict")

def predict(data: dict):

    df = pd.DataFrame([data])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    prediction = model.predict(df)

    return {"predicted_usage": float(prediction[0])}