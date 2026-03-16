from pydantic import BaseModel, Field


class IntroductionRequest(BaseModel):
    name: str = Field(..., description="Name")
    message: str = Field(..., description="Message")


class IntroductionResponse(BaseModel):
    message: str = Field(..., description="Message")
