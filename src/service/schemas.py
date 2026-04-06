from enum import Enum

from pydantic import BaseModel, Field


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


class ContactEnum(str, Enum):
    cell = "cell"
    email = "email"


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
    campaign: int = Field(..., ge=1)
    job: JobEnum
    marital: MaritalEnum
    education: EducationEnum
    previous: int = Field(..., ge=0)
    month: str
    pdays: int = Field(..., ge=-1)
    poutcome: PreviousCampaignOutcomeEnum
    contact: ContactEnum

    def to_model_input(self) -> dict:
        import math
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        return {
            "age": self.age,
            "balance": self.balance,
            "negative_balance": int(self.balance < 0),
            "default": int(self.default),
            "housing": int(self.housing),
            "loan": int(self.loan),
            "campaign": math.log1p(self.campaign),
            "job": self.job.value,
            "marital": self.marital.value,
            "education": self.education.value,
            "month": month_map.get(self.month.lower(), 5),
            "was_contacted": int(self.pdays != -1),
            "pdays": self.pdays,
            "previous": self.previous,
            "poutcome": self.poutcome.value,
            "contact": self.contact.value,
        }


class PredictionResponse(BaseModel):
    prediction: float
