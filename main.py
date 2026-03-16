from fastapi import FastAPI

from src.models import IntroductionRequest, IntroductionResponse

app = FastAPI()


@app.get("/health", tags=["Infrastructure"])
async def health():
    """Kubernetes liveness / readiness probe."""
    return {"status": "healthy"}


@app.post("/intro", response_model=IntroductionResponse, tags=["Introduction"])
def intro(request: IntroductionRequest):
    return IntroductionResponse(message=f"Hello {request.name}, {request.message}")
