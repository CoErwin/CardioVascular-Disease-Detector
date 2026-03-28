import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("../data/combined_dataset.csv")

X = df[["age","chol","trestbps","thalach","oldpeak"]]

def label_workout(row):
    if row["target"] == 1:
        return [0,0,1]  # light
    elif row["age"] > 50:
        return [1,1,1]  # moderate
    else:
        return [2,2,2]  # intense

y = np.array(df.apply(label_workout, axis=1).tolist())
y = [to_categorical(y[:,i], num_classes=3) for i in range(3)]

model = Sequential([
    Input(shape=(5,)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(9, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(X, np.concatenate(y, axis=1), epochs=20)

model.save("../model/workout_nn.h5")

print("✅ Workout NN trained")