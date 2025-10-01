# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Student Exam Pass Predictor", layout="centered")

# --- Add an image banner ---
st.image("student_exam.jpg", use_column_width=True, caption="📘 Student Exam Predictor")

st.title("📘 Student Exam Pass Predictor")
st.write("Enter student info and click **Predict**")

# --- Load artifacts ---
@st.cache_data(show_spinner=False)
def load_artifacts():
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("parental_le.pkl")
    # If you saved features.json:
    try:
        import json
        with open("features.json","r") as f:
            features = json.load(f)
    except:
        features = ["Hours_Studied","Attendance","Previous_Scores","Parental_Involvement","Sleep_Hours"]
    return model, scaler, le, features

model, scaler, le, FEATURES = load_artifacts()

# Show encoder classes for parental involvement
parent_options = list(le.classes_) if hasattr(le, "classes_") else ["Low","Medium","High"]

# --- Sidebar or main inputs ---
st.sidebar.header("Input student features")
hours = st.sidebar.number_input("Hours Studied (per week)", min_value=0, max_value=200, value=10)
attendance = st.sidebar.number_input("Attendance (%)", min_value=0, max_value=100, value=90)
previous = st.sidebar.number_input("Previous Scores (avg)", min_value=0, max_value=100, value=60)
parental = st.sidebar.selectbox("Parental Involvement", parent_options, index=0)
sleep = st.sidebar.number_input("Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

# Build feature vector in the same order used for training
try:
    parental_enc = int(le.transform([str(parental)])[0])
except Exception as e:
    parental_enc = 0

X_input = np.array([[hours, attendance, previous, parental_enc, sleep]], dtype=float)

# Scale using saved scaler
X_scaled = scaler.transform(X_input)

if st.button("Predict"):
    prob = model.predict_proba(X_scaled)[0,1]
    pred = model.predict(X_scaled)[0]
    st.metric("Predicted Class", "Pass" if pred==1 else "Fail")
    st.write(f"Probability of Passing: **{prob*100:.2f}%**")

    st.write("---")
    st.write("**Note:** Model was trained on historical data. Use responsibly.")
