import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.service.schemas import PredictionRequest, PredictionResponse
from src.service.utils import EXPECTED_COLS

mlflow.set_tracking_uri("https://mlflow-server-production-c6e7.up.railway.app/")

model_name = "KNN"
alias = "KNN_knn_bmd_tuned"
model_uri = f"models:/{model_name}@{alias}"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI()


@app.get("/health", tags=["Infrastructure"])
async def health():
    """Kubernetes liveness / readiness probe."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    df = pd.DataFrame([request.to_model_input()])
    df = df.reindex(columns=EXPECTED_COLS, fill_value=0)
    predictions: np.ndarray = model.predict(df)  # ty: ignore[invalid-assignment]
    prediction = "likely" if predictions[0] else "unlikely"
    return PredictionResponse(prediction=prediction)
