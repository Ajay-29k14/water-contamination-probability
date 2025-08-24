# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

MODEL_PATH = "models/model.joblib"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found: {MODEL_PATH} - run train_model.py first")

bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
COLUMNS = bundle["cols"]  # ["rainfall_3h","temperature","humidity","turbidity"]

app = FastAPI(title="Contamination Risk API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite dev URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    rainfall: float       # rainfall sum over window (mm)
    temperature: float    # °C
    humidity: float       # %
    turbidity: float      # NTU

class PredictResponse(BaseModel):
    contamination_probability: float  # percent 0-100

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        x = np.array([[req.rainfall, req.temperature, req.humidity, req.turbidity]])
        prob = MODEL.predict_proba(x)[0][1]
        return {"contamination_probability": round(float(prob*100), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
