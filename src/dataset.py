import pandas as pd
from preprocess import scale_data, create_sequences

def load_data(path, window=5):
    df = pd.read_csv(path)
    values = df["value"].values

    scaled = scale_data(values)
    X, y = create_sequences(scaled, window)

    return X, y