from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.model_service import predict_risk
from app.services.report_ai import get_report

# KEEP THESE (as you asked)
from app.services.diet_ai import get_diet_recommendation
from app.services.workout_ai import get_workout_recommendation

app = FastAPI()

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== ROOT =====================
@app.get("/")
def home():
    return {"message": "API running"}

# ===================== PREDICT =====================
@app.post("/predict")
async def predict(data: dict):
    try:
        result = predict_risk(data)

        # ✅ ALWAYS RETURN CLEAN RESPONSE
        return {
            "risk": float(result.get("risk", 0))
        }

    except Exception as e:
        print("❌ API Error:", e)

        # ✅ NEVER BREAK FRONTEND
        return {
            "risk": 0.0
        }

# ===================== DIET =====================
@app.get("/diet")
def diet(risk: float = 50, diet_preference: str = "veg"):
    try:
        return get_diet_recommendation({
            "risk": risk,
            "diet_preference": diet_preference   # ← pass from query param
        })
    except Exception as e:
        print("❌ Diet error:", e)
        return {"breakfast": [], "lunch": [], "dinner": []}

# ===================== WORKOUT =====================
@app.get("/workout")
def workout(risk: float = 50, age: int = 35):
    try:
        return get_workout_recommendation({
            "risk": risk,
            "age" : age
        })
    except Exception as e:
        print("❌ Workout error:", e)
        return {
            "cardio"      : [{"name": "Walking",  "duration": "20 mins"}],
            "strength"    : [{"name": "Pushups",  "sets": 3, "reps": 10}],
            "flexibility" : [{"name": "Yoga",     "duration": "15 mins"}],
        }


# ======================Report======================
@app.post("/report")
async def report(data: dict):
    try:
        return get_report(data)
    except Exception as e:
        print("❌ Report API Error:", e)
        return {"risk": 0, "advice": "Error generating report."}