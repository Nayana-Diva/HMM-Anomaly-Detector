import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="HMM Anomaly Detector",
    page_icon="📊",
    layout="wide"
)

st.title("HMM-Based Machine Anomaly Detector")
st.caption("Gaussian Hidden Markov Model | Machine Sensor Monitoring")

try:
    df = pd.read_csv("outputs/detection_results.csv")
    package = joblib.load("models/hmm_model.pkl")
except FileNotFoundError:
    st.error("Run main.py first to generate the detection results.")
    st.stop()

actual = df["actual_anomaly"]
predicted = df["predicted_anomaly"]

accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted, zero_division=0)
recall = recall_score(actual, predicted, zero_division=0)
f1 = f1_score(actual, predicted, zero_division=0)

st.subheader("Detection Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Precision", f"{precision:.2%}")
c3.metric("Recall", f"{recall:.2%}")
c4.metric("F1 Score", f"{f1:.2%}")

st.divider()

st.subheader("Sensor Monitoring")

sensor = st.selectbox(
    "Select sensor",
    ["temperature", "vibration", "pressure"]
)

fig = px.line(
    df,
    y=sensor,
    title=f"{sensor.title()} Sensor Readings",
    labels={"index": "Reading", "y": sensor}
)

actual_points = df[df["actual_anomaly"] == 1]
predicted_points = df[df["predicted_anomaly"] == 1]

fig.add_scatter(
    x=actual_points.index,
    y=actual_points[sensor],
    mode="markers",
    name="Actual anomalies",
    marker=dict(color="orange", size=9, symbol="circle")
)

fig.add_scatter(
    x=predicted_points.index,
    y=predicted_points[sensor],
    mode="markers",
    name="Detected anomalies",
    marker=dict(color="red", size=7, symbol="x")
)

fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)
col1.metric("Hidden States", package["states"])
col2.metric("Window Size", package["window"])
col3.metric("Anomaly Threshold", f'{package["threshold"]:.4f}')

st.subheader("Detection Results")

show_all = st.checkbox("Show all sensor readings", value=False)

if show_all:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(
        df[df["predicted_anomaly"] == 1],
        use_container_width=True
    )
