# train_model.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def generate_synthetic(n=3000, seed=42):
    rng = np.random.RandomState(seed)
    rainfall = rng.exponential(scale=30.0, size=n)      # mm
    temperature = rng.normal(loc=28, scale=5, size=n)   # °C
    humidity = rng.normal(loc=70, scale=12, size=n)     # %
    turbidity = rng.exponential(scale=50, size=n)       # NTU

    # simple heuristic -> probability score
    score = (
        0.45 * (rainfall / (rainfall.max() + 1)) +
        0.30 * (turbidity / (turbidity.max() + 1)) +
        0.15 * (humidity / 100.0) +
        0.10 * ((temperature - 10) / 30.0)
    )
    prob = 1 / (1 + np.exp(-6 * (score - 0.35)))  # logistic mapping
    labels = (rng.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "rainfall_3h": rainfall,
        "temperature": temperature,
        "humidity": humidity,
        "turbidity": turbidity,
        "label": labels
    })
    return df

def train_and_save(output_path="models/model.joblib"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic()
    X = df[["rainfall_3h","temperature","humidity","turbidity"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "cols": X.columns.tolist()}, output_path)
    print("Model trained and saved to", output_path)
    # optional: print test set score
    print("Train accuracy (approx):", model.score(X_train, y_train))
    print("Test accuracy (approx):", model.score(X_test, y_test))

if __name__ == "__main__":
    train_and_save()
