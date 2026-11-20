import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

FEATURES = ["temperature", "vibration", "pressure"]
WINDOW = 10
STEP = 1
PERCENTILE = 1


def get_window_scores(model, data):
    scores = []

    for start in range(0, len(data) - WINDOW + 1, STEP):
        window = data[start:start + WINDOW]
        scores.append(model.score(window) / len(window))

    return np.asarray(scores)


def train_model():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/normal_sensor_data.csv")
    X = df[FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model = None
    best_bic = float("inf")
    best_states = None

    for states in [2, 3, 4, 5]:
        model = GaussianHMM(
            n_components=states,
            covariance_type="diag",
            n_iter=300,
            tol=0.01,
            random_state=42,
            min_covar=0.001
        )

        model.fit(X_scaled)
        likelihood = model.score(X_scaled)
        dimensions = X_scaled.shape[1]

        parameters = (
            states * (states - 1)
            + (states - 1)
            + 2 * states * dimensions
        )

        bic = -2 * likelihood + parameters * np.log(len(X_scaled))

        print(
            f"States: {states} | "
            f"Log likelihood: {likelihood:.2f} | "
            f"BIC: {bic:.2f} | "
            f"Converged: {model.monitor_.converged}"
        )

        if np.isfinite(bic) and bic < best_bic:
            best_bic = bic
            best_model = model
            best_states = states

    if best_model is None:
        raise RuntimeError("HMM training failed.")

    train_scores = get_window_scores(best_model, X_scaled)
    threshold = float(np.percentile(train_scores, PERCENTILE))

    package = {
        "model": best_model,
        "scaler": scaler,
        "threshold": threshold,
        "states": best_states,
        "features": FEATURES,
        "window": WINDOW,
        "step": STEP,
        "percentile": PERCENTILE
    }

    joblib.dump(package, "models/hmm_model.pkl")

    print("\nMODEL SELECTION")
    print("================")
    print("Selected hidden states:", best_states)
    print("Best BIC:", round(best_bic, 4))
    print("Threshold:", round(threshold, 4))
    print("Model saved successfully.")

    return package


if __name__ == "__main__":
    train_model()
