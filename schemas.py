from pydantic import BaseModel, Field


class UserInput(BaseModel):
    # Basic Info
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., description="0 = Female, 1 = Male")

    # Chest pain type
    cp: int = Field(..., description="0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")

    # Vitals
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Cholesterol level")

    # Binary flags
    fbs: int = Field(..., description="Fasting blood sugar >120 (1=True, 0=False)")
    restecg: int = Field(..., description="0: Normal, 1: ST-T abnormality, 2: Hypertrophy")

    # Heart metrics
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: int = Field(..., description="Exercise angina (1=Yes, 0=No)")
    oldpeak: float = Field(..., description="ST depression")

    # Slope
    slope: int = Field(..., description="0: Up, 1: Flat, 2: Down")

    # Major vessels
    ca: int = Field(..., ge=0, le=4)

    # Thalassemia
    thal: int = Field(..., description="1: Normal, 2: Fixed, 3: Reversible")

    # Extra inputs (for AI modules)
    weight: float = Field(..., gt=0)
    diet_type: str = Field(..., description="veg / nonveg")



# ✅ Optional response schema (good practice)
class PredictionResponse(BaseModel):
    risk: int
    probability: float