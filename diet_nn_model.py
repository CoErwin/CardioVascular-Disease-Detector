import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
import os

# Load heart dataset
df = pd.read_csv("../data/combined_dataset.csv")

X = df[["age","chol","trestbps","thalach","oldpeak"]]

# Create labels based on health conditions
def label_diet(row):
    if row["chol"] > 240 or row["trestbps"] > 140:
        return [0,0,0]  # strict diet
    elif row["age"] > 50:
        return [1,1,1]  # moderate
    else:
        return [2,2,2]  # flexible

y = np.array(df.apply(label_diet, axis=1).tolist())

# One-hot encoding
y = [to_categorical(y[:,i], num_classes=3) for i in range(3)]

# Model
model = Sequential([
    Input(shape=(5,)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(9, activation='softmax')  # 3 outputs merged
])

# Split outputs
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy'
)

model.fit(X, np.concatenate(y, axis=1), epochs=20)

model.save("../model/diet_nn.h5")

print("✅ Diet NN trained")