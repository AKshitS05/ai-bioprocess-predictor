import streamlit as st
import pandas as pd
import joblib


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Bioprocess Predictor",
    layout="wide"
)


# =========================
# LOAD MODEL
# =========================

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# =========================
# TITLE
# =========================

st.title("AI-Driven Bioprocess Predictor")

st.write(
    "Predict Penicillin Concentration using Fermentation Parameters"
)


# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("Enter Process Parameters")


# Store inputs
input_data = {}


# Auto-create inputs from saved features
for feature in feature_columns:

    default_value = 0.0

    if "Temperature" in feature:
        default_value = 298.0

    elif "pH" in feature:
        default_value = 7.0

    elif "oxygen" in feature.lower():
        default_value = 5.0

    elif "RPM" in feature:
        default_value = 100.0

    elif "rate" in feature.lower():
        default_value = 10.0

    value = st.sidebar.number_input(
        label=feature,
        value=float(default_value),
        step=0.1
    )

    input_data[feature] = value


# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Penicillin Concentration"):

    try:

        # Create dataframe
        input_df = pd.DataFrame([input_data])

        # VERY IMPORTANT
        # Match exact training order
        input_df = input_df[feature_columns]

        # Prediction
        prediction = model.predict(input_df)[0]

        st.success("Prediction Successful")

        st.metric(
            label="Predicted Penicillin Concentration",
            value=f"{prediction:.4f} g/L"
        )

        st.subheader("Input Summary")

        st.dataframe(input_df)

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)