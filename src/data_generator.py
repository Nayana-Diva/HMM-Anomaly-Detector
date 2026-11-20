import numpy as np
import pandas as pd
import os


def generate_data(n, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    temperature = 50 + 2*np.sin(t/40) + rng.normal(0, 0.8, n)
    vibration = 5 + 0.5*np.sin(t/20) + rng.normal(0, 0.2, n)
    pressure = 100 + 3*np.sin(t/50) + rng.normal(0, 1, n)

    return pd.DataFrame({
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure
    })


def generate_datasets():
    os.makedirs("data", exist_ok=True)

    normal = generate_data(1000, seed=42)
    test = generate_data(500, seed=123)

    test["actual_anomaly"] = 0

    anomaly_ranges = [(80, 90), (200, 210), (320, 330), (420, 430)]

    for start, end in anomaly_ranges:
        test.loc[start:end-1, "temperature"] += 15
        test.loc[start:end-1, "vibration"] += 4
        test.loc[start:end-1, "pressure"] -= 20
        test.loc[start:end-1, "actual_anomaly"] = 1

    normal.to_csv("data/normal_sensor_data.csv", index=False)
    test.to_csv("data/test_sensor_data.csv", index=False)

    print("Sensor datasets generated successfully.")
    print("Normal records:", len(normal))
    print("Test records:", len(test))
    print("Actual anomaly records:", int(test["actual_anomaly"].sum()))

    return normal, test
