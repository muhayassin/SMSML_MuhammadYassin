import os
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
TFJS_DIR = os.path.join(BASE_DIR, "submission", "tfjs_model")

os.makedirs(TFJS_DIR, exist_ok=True)

# ===============================
# MLFLOW SETUP
# ===============================
mlflow.set_experiment("Telco-Customer-Churn")

# ===============================
# LOAD DATASET
# ===============================
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ===============================
# PREPROCESSING
# ===============================
print("[INFO] Preprocessing data...")

df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

categorical_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ===============================
# FEATURE & TARGET
# ===============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# BUILD MODEL
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAIN + LOG MLFLOW
# ===============================
with mlflow.start_run():
    print("[INFO] Training model...")

    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 32)

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    print("[INFO] Evaluating model...")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("test_accuracy", acc)

    print(f"Test Accuracy: {acc:.4f}")

    print("[INFO] Saving model...")
    model.save(SAVED_MODEL_DIR)

    mlflow.tensorflow.log_model(model, artifact_path="model")

print("[INFO] Converting model to TensorFlow.js...")
os.system(
    f"tensorflowjs_converter --input_format=tf_saved_model {SAVED_MODEL_DIR} {TFJS_DIR}"
)

print("[SUCCESS] Modelling selesai. Model siap untuk submission ASAH.")
