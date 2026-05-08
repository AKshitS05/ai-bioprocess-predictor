import os
import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# =========================
# AUTO TRAIN FUNCTION
# =========================

def train_model():

    df = pd.read_csv("batch-1-10.csv")

    remove_columns = [
        "batch_id",
        "Fault reference(Fault_ref:Fault ref)",
        "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)"
    ]

    for col in remove_columns:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    TARGET_COLUMN = "Penicillin concentration(P:g/L)"

    df = df.dropna(subset=[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    feature_columns = X.columns.tolist()

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "model.pkl")
    joblib.dump(feature_columns, "feature_columns.pkl")


# =========================
# TRAIN IF MODEL MISSING
# =========================

if not os.path.exists("model.pkl"):

    train_model()


# =========================
# LOAD MODEL
# =========================

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Bioprocess Predictor",
    layout="wide"
)


# =========================
# TITLE
# =========================

st.title("AI-Driven Bioprocess Predictor")

st.write(
    "Predict Penicillin Concentration using Fermentation Parameters"
)


# =========================
# SIDEBAR
# =========================

st.sidebar.header("Input Parameters")


input_data = {}

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
# PREDICT
# =========================

if st.button("Predict Penicillin Concentration"):

    try:

        input_df = pd.DataFrame([input_data])

        input_df = input_df[feature_columns]

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