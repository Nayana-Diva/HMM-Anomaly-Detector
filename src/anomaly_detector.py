import numpy as np
import joblib


def detect_anomalies(data):
    package = joblib.load("models/hmm_model.pkl")

    features = package["features"]
    window_size = package["window"]
    step = package["step"]
    threshold = package["threshold"]

    X = data[features].values
    X_scaled = package["scaler"].transform(X)

    if len(X) < window_size:
        raise ValueError(
            f"At least {window_size} readings are required."
        )

    scores = np.full(len(X), np.nan)

    for start in range(0, len(X) - window_size + 1, step):
        end = start + window_size
        window = X_scaled[start:end]

        score = package["model"].score(window) / window_size

        centre = start + window_size // 2
        scores[centre] = score

    valid = np.where(~np.isnan(scores))[0]

    scores[:valid[0]] = scores[valid[0]]
    scores[valid[-1] + 1:] = scores[valid[-1]]

    anomaly_flags = (scores < threshold).astype(int)

    return anomaly_flags, scores
