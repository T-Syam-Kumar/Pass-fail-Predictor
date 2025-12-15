import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Predict whether a student will **Pass or Fail** with probability")

st.divider()

# =========================
# 🔹 Manual Input Section
# =========================
st.subheader("🔹 Manual Input")

study_hours = st.slider("Study Hours per Day", 0, 10, 3)
attendance = st.slider("Attendance Percentage", 0, 100, 70)
previous_score = st.slider("Previous Exam Score", 0, 100, 50)

if st.button("Predict Result"):
    input_data = np.array([[study_hours, attendance, previous_score]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ Student is likely to PASS")
    else:
        st.error(f"❌ Student is likely to FAIL")

    st.info(f"📊 **Pass Probability:** {probability:.2f}%")

st.divider()

# =========================
# 🔹 CSV Upload Section
# =========================
st.subheader("📂 Upload CSV File")

st.write("CSV must have columns:")
st.code("study_hours, attendance, previous_score")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📄 Uploaded Data:")
    st.dataframe(df)

    if st.button("Predict for Uploaded Data"):
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1] * 100

        df["Prediction"] = ["Pass" if p == 1 else "Fail" for p in predictions]
        df["Pass Probability (%)"] = probabilities.round(2)

        st.success("✅ Predictions Completed")
        st.dataframe(df)
