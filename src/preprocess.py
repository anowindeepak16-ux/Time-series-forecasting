import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def scale_data(data):
    return scaler.fit_transform(data.reshape(-1, 1))

def inverse_scale(data):
    return scaler.inverse_transform(data)

def create_sequences(data, window=5):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)