import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="HMM Anomaly Detector", layout="wide")
st.title("Machine Sensor Anomaly Detection")
st.caption("Gaussian Hidden Markov Model | Simulated sensor stream")

try:
    df = pd.read_csv("outputs/detection_results.csv")
    package = joblib.load("models/hmm_model.pkl")
except FileNotFoundError:
    st.error("Run the main.py file first to generate detection results.")
    st.stop()

total = len(df)
detected = int(df["predicted_anomaly"].sum())
actual = int(df["actual_anomaly"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Total readings", total)
c2.metric("Detected anomaly readings", detected)
c3.metric("Actual anomaly readings", actual)

sensor = st.selectbox(
    "Select sensor",
    ["temperature", "vibration", "pressure"]
)

fig = px.line(df, y=sensor, title=f"{sensor.title()} Sensor Readings")

anomalies = df[df["predicted_anomaly"] == 1]

fig.add_scatter(
    x=anomalies.index,
    y=anomalies[sensor],
    mode="markers",
    name="Detected anomalies",
    marker=dict(color="red", size=7)
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Detection Results")
st.dataframe(
    df[df["predicted_anomaly"] == 1],
    use_container_width=True
)

st.subheader("Model Information")
st.write("Model: Gaussian Hidden Markov Model")
st.write("Selected hidden states:", package["states"])
st.write("Anomaly threshold:", round(package["threshold"], 4))
st.write("Window size:", package["window"])
