from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class JobEnum(str, Enum):
    admin = "admin"
    blue_collar = "blue-collar"
    entrepreneur = "entrepreneur"
    housemaid = "housemaid"
    management = "management"
    retired = "retired"
    self_employed = "self-employed"
    services = "services"
    student = "student"
    technician = "technician"
    unemployed = "unemployed"
    unknown = "unknown"


class MaritalEnum(str, Enum):
    married = "married"
    single = "single"
    divorced = "divorced"


class EducationEnum(str, Enum):
    primary = "primary"
    secondary = "secondary"
    tertiary = "tertiary"
    unknown = "unknown"


class PreviouslyContactedEnum(str, Enum):
    never = "never"
    within_a_week = "within_a_week"
    within_a_month = "within_a_month"
    over_a_month = "over_a_month"


class PreviousCampaignOutcomeEnum(str, Enum):
    failure = "failure"
    other = "other"
    success = "success"
    unknown = "unknown"


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    balance: float
    default: bool
    housing: bool
    loan: bool
    campaign: int
    previous_campaign: int
    job: JobEnum
    marital: MaritalEnum
    education: EducationEnum
    pcontacted: PreviouslyContactedEnum
    poutcome: PreviousCampaignOutcomeEnum

    def to_model_input(self) -> dict:
        # In hindsight, I should have renamed the columns in the cvs file
        return {
            "age": self.age,
            "job": self.job.value,
            "marital": self.marital.value,
            "education": self.education.value,
            "default": "yes" if self.default else "no",
            "balance": self.balance,
            "housing": int(self.housing),
            "loan": int(self.loan),
            "campaign": self.campaign,
            "pcontacted": self.pcontacted.value,
            "previous": self.previous_campaign,
            "poutcome": self.poutcome.value,
        }


class PredictionResponse(BaseModel):
    prediction: str
