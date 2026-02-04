import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

from dataset import load_data

BASE_DIR = r"D:\time_series"
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.h5")

# Load model safely
model = load_model(MODEL_PATH, compile=False)

def integrated_gradients_lstm(
    model,
    x,
    baseline=None,
    steps=50
):
    """
    x shape: (1, timesteps, features)
    """

    x = tf.cast(x, tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(x)
    else:
        baseline = tf.cast(baseline, tf.float32)

    grads_list = []

    for alpha in tf.linspace(0.0, 1.0, steps):
        interpolated = baseline + alpha * (x - baseline)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = model(interpolated)

            # If output is (1, 1) or (1,)
            pred = tf.reduce_sum(pred)

        grads = tape.gradient(pred, interpolated)
        grads_list.append(grads)

    grads = tf.stack(grads_list)

    # Trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    ig = (x - baseline) * avg_grads
    return ig.numpy()