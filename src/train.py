import os
from dataset import load_data
from model import build_model

X, y = load_data("data/energy.csv")

model = build_model((X.shape[1], 1))

model.fit(X, y, epochs=20, batch_size=8)

os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")

print("✅ Model saved to models/lstm_model.h5")