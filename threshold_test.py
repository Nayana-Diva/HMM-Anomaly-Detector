import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("outputs/detection_results.csv")
package = joblib.load("models/hmm_model.pkl")

scores = df["log_likelihood"].values
actual = df["actual_anomaly"].values

print("THRESHOLD COMPARISON")
print("-" * 65)

for percentile in [1, 2, 5, 10, 15, 20, 25, 30]:
    threshold = np.percentile(scores, percentile)
    predicted = (scores < threshold).astype(int)

    print(
        f"Percentile {percentile:2d} | "
        f"Threshold {threshold:8.3f} | "
        f"Precision {precision_score(actual, predicted, zero_division=0):.3f} | "
        f"Recall {recall_score(actual, predicted, zero_division=0):.3f} | "
        f"F1 {f1_score(actual, predicted, zero_division=0):.3f} | "
        f"Detected {predicted.sum()}"
    )
