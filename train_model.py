import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("batch-1-10.csv")

print("Dataset Loaded Successfully")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================
# REMOVE USELESS COLUMNS
# =========================

remove_columns = [
    "batch_id",
    "Fault reference(Fault_ref:Fault ref)",
    "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)"
]

for col in remove_columns:
    if col in df.columns:
        df.drop(columns=col, inplace=True)


# =========================
# TARGET COLUMN
# =========================

TARGET_COLUMN = "Penicillin concentration(P:g/L)"


# =========================
# REMOVE EMPTY TARGET ROWS
# =========================

df = df.dropna(subset=[TARGET_COLUMN])


# =========================
# FEATURES + TARGET
# =========================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# =========================
# SAVE FEATURE ORDER
# =========================

feature_columns = X.columns.tolist()

joblib.dump(feature_columns, "feature_columns.pkl")

print("\nFeatures Used:")
for col in feature_columns:
    print(col)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# CREATE PIPELINE
# =========================

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


# =========================
# TRAIN MODEL
# =========================

print("\nTraining Model...")

pipeline.fit(X_train, y_train)

print("Training Complete")


# =========================
# EVALUATION
# =========================

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print(f"MAE : {mae:.4f}")
print(f"R2 Score : {r2:.4f}")


# =========================
# SAVE MODEL
# =========================

joblib.dump(pipeline, "model.pkl")

print("\nModel Saved Successfully")
print("Generated Files:")
print("- model.pkl")
print("- feature_columns.pkl")