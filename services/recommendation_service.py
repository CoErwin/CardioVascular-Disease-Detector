from app.services.calorie_service import calculate_calories

def generate_recommendations(risk_percentage, data):

    calories = calculate_calories(
        data.weight, data.height, data.age, data.sex
    )

    # -----------------------------------
    # Macro Distribution
    # -----------------------------------
    if risk_percentage > 70:
        macros = {"protein": 30, "carbs": 40, "fats": 30}
    elif 40 <= risk_percentage <= 70:
        macros = {"protein": 25, "carbs": 45, "fats": 30}
    else:
        macros = {"protein": 25, "carbs": 50, "fats": 25}

    # -----------------------------------
    # Diet Type Logic
    # -----------------------------------
    if data.diet_type == "veg":
        breakfast = {
            "main": "Oatmeal with chia seeds",
            "protein": "Paneer scramble or sprouts",
            "fruit": "1 seasonal fruit",
            "drink": "Green tea"
        }

        lunch = {
            "carb": "2 multigrain rotis or brown rice",
            "protein": "Dal + Rajma/Chole",
            "veg": "Boiled vegetables",
            "salad": "Cucumber + carrot salad"
        }

        dinner = {
            "main": "Vegetable soup",
            "protein": "Grilled paneer/tofu",
            "side": "Light salad"
        }

    else:
        breakfast = {
            "main": "Oats with boiled eggs",
            "protein": "2 egg whites + 1 whole egg",
            "fruit": "1 apple/banana",
            "drink": "Black coffee/Green tea"
        }

        lunch = {
            "carb": "Brown rice",
            "protein": "Grilled chicken/fish (100–150g)",
            "veg": "Steamed vegetables",
            "salad": "Fresh salad bowl"
        }

        dinner = {
            "main": "Clear chicken/veg soup",
            "protein": "Grilled fish or boiled eggs",
            "side": "Sauteed vegetables"
        }

    diet = {
        "calories": calories,
        "macros": macros,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner
    }

    # -----------------------------------
    # WORKOUT (Improved)
    # -----------------------------------
    if risk_percentage > 70:
        workout_plan = {
            "cardio": "30 mins brisk walking",
            "strength": "Wall pushups (3x10), bodyweight squats",
            "flexibility": "10 mins yoga/stretching",
            "frequency": "5 days/week",
            "note": "Avoid high intensity."
        }
        workout_distribution = {"cardio": 60, "strength": 25, "flexibility": 15}

    elif 40 <= risk_percentage <= 70:
        workout_plan = {
            "cardio": "Jogging or cycling (25 mins)",
            "strength": "Pushups, planks, light dumbbells",
            "flexibility": "Yoga 15 mins",
            "frequency": "4–5 days/week",
            "note": "Maintain moderate intensity."
        }
        workout_distribution = {"cardio": 50, "strength": 35, "flexibility": 15}

    else:
        workout_plan = {
            "cardio": "Running, swimming or sports",
            "strength": "Pushups, pullups, weight training",
            "flexibility": "Dynamic stretching",
            "frequency": "4–6 days/week",
            "note": "Maintain consistency."
        }
        workout_distribution = {"cardio": 40, "strength": 40, "flexibility": 20}

    workout = {
        "plan": workout_plan,
        "distribution": workout_distribution
    }

    return diet, workout