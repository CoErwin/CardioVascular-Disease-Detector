import pandas as pd

files = [
    "../data/food/FOOD-DATA-GROUP1.csv",
    "../data/food/FOOD-DATA-GROUP2.csv",
    "../data/food/FOOD-DATA-GROUP3.csv",
    "../data/food/FOOD-DATA-GROUP4.csv",
    "../data/food/FOOD-DATA-GROUP5.csv",
]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

df.to_csv("final_food_dataset.csv", index=False)

print("✅ Combined food dataset created")