from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate(actual, predicted):
    print("\nEVALUATION RESULTS")
    print("--------------------------")
    print("Accuracy :", round(accuracy_score(actual, predicted), 4))
    print("Precision:", round(precision_score(actual, predicted, zero_division=0), 4))
    print("Recall   :", round(recall_score(actual, predicted, zero_division=0), 4))
    print("F1 Score :", round(f1_score(actual, predicted, zero_division=0), 4))
    print("Detected anomalies:", int(sum(predicted)))
    print("Actual anomalies  :", int(sum(actual)))
