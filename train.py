
"""
NASA Near-Earth Object (NEO) Classification
Model Training and Production Model Saving

Algorithm: Random Forest Classifier
Target: hazardous
Output: model_prod.pkl

Run:
    python train.py neo.csv

If no filename is provided:
    python train.py
    (Loads neo.csv from the project folder)
"""

from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

DATA_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else BASE_DIR / "neo.csv"
)

MODEL_PATH = BASE_DIR / "model_prod.pkl"

FEATURES = [
    "absolute_magnitude",
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance"
]

TARGET = "hazardous"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("\nLoading NASA NEO dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place neo.csv in the project folder or "
        "provide the correct CSV path."
    )

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

print("\nCleaning dataset...")

df = df[required_columns].copy()

# Convert numerical columns
for col in FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Handle infinite values
df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

# Convert target to binary
def convert_target(value):

    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    value = str(value).strip().lower()

    if value in ["true", "1", "yes"]:
        return 1

    if value in ["false", "0", "no"]:
        return 0

    return np.nan


df[TARGET] = df[TARGET].apply(convert_target)

# Remove records with missing target
df = df.dropna(subset=[TARGET])

df[TARGET] = df[TARGET].astype(int)

# Remove records where all input features are missing
df = df.dropna(subset=FEATURES, how="all")

X = df[FEATURES]
y = df[TARGET]

print("Cleaned dataset shape:", df.shape)

print("\nTarget distribution:")
print(y.value_counts())

if y.nunique() != 2:
    raise ValueError(
        "Target must contain both classes: 0 and 1."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "Each class needs at least two records."
    )

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# BUILD MODEL PIPELINE
# --------------------------------------------------

print("\nBuilding Random Forest model...")

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Hazardous",
            "Hazardous"
        ],
        zero_division=0
    )
)

# --------------------------------------------------
# RETRAIN ON COMPLETE DATASET
# --------------------------------------------------

print("\nRetraining model using complete dataset...")

model.fit(X, y)

# --------------------------------------------------
# SAVE PRODUCTION MODEL
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

print("\n========================================")
print("PRODUCTION MODEL SAVED SUCCESSFULLY")
print("========================================")

print("Model file:", MODEL_PATH)
print("Model type: Random Forest Classifier")
print("Input features:", FEATURES)
print("Target:", TARGET)

# --------------------------------------------------
# TEST SAVED MODEL
# --------------------------------------------------

print("\nTesting saved model...")

loaded_model = joblib.load(MODEL_PATH)

sample = X.iloc[[0]]

prediction = loaded_model.predict(sample)[0]

probability = loaded_model.predict_proba(sample)[0]

print("Sample prediction:", prediction)
print(
    "Hazardous probability:",
    round(probability[1] * 100, 2),
    "%"
)

print("\nModel is ready for Streamlit application!")
