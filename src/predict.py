import os
import numpy as np
from tensorflow.keras.models import load_model
from dataset import load_data
from preprocess import inverse_scale

BASE_DIR = r"D:\time_series"
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.h5")

model = load_model(MODEL_PATH, compile=False)

if __name__ == "__main__":
    X, y = load_data("data/energy.csv")

    pred = model.predict(X)

    pred = inverse_scale(pred)
    y = inverse_scale(y.reshape(-1, 1))

    print("Predictions (first 5):")
    print(pred[:5])