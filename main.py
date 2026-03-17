import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI

from src.service.schemas import PredictionRequest, PredictionResponse
from src.service.utils import EXPECTED_COLS

model_name = "NeuralNetwork"
alias = "NeuralNetwork_nn_bmd_tuned"

model_uri = f"models:/{model_name}@{alias}"
mlflow.set_tracking_uri("https://mlflow-server-production-c6e7.up.railway.app/")

# Load the model directly for inference
loaded_model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI()


@app.get("/health", tags=["Infrastructure"])
async def health():
    """Kubernetes liveness / readiness probe."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    df = pd.DataFrame([request.to_model_input()])
    cat_features = df.select_dtypes(exclude=["number"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_features, drop_first=False, dtype=int)
    df = df.reindex(columns=EXPECTED_COLS, fill_value=0)
    predictions: np.ndarray = loaded_model.predict(df)  # ty: ignore[invalid-assignment]
    prediction = "likely" if predictions[0] else "unlikely"
    return PredictionResponse(prediction=prediction)
