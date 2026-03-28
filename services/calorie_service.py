def calculate_calories(weight, height, age, sex):
    if sex == 1:  # male
        bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:  # female
        bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

    daily_calories = bmr * 1.3
    return round(daily_calories, 2)