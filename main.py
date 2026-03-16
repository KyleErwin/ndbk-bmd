from fastapi import FastAPI
from src.models import IntroductionRequest, IntroductionResponse

app = FastAPI()


@app.post("/intro", response_model=IntroductionResponse, tags=["Introduction"])
def hello(request: IntroductionRequest):
    return IntroductionResponse(message=f"Hello {request.name}, {request.message}")
