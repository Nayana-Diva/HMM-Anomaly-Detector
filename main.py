import os

from src.data_generator import generate_datasets
from src.train_model import train_model
from src.anomaly_detector import detect_anomalies
from src.evaluation import evaluate


def main():
    print("\nHMM MACHINE ANOMALY DETECTOR")
    print("============================")

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("\n1. Generating sensor data...")
    normal, test = generate_datasets()

    print("\n2. Training Hidden Markov Model...")
    train_model()

    print("\n3. Detecting anomalies...")
    flags, scores = detect_anomalies(test)

    results = test.copy()
    results["predicted_anomaly"] = flags
    results["log_likelihood"] = scores

    results.to_csv("outputs/detection_results.csv", index=False)

    print("\n4. Evaluating detection performance...")
    evaluate(test["actual_anomaly"], flags)

    print("\n5. Results saved to outputs/detection_results.csv")
    print("\nPROJECT EXECUTION COMPLETED.")


if __name__ == "__main__":
    main()
