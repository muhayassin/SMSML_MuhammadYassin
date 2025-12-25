import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ===============================
# SET MLFLOW EXPERIMENT
# ===============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Telco-Customer-Churn-RF")

# ===============================
# LOAD DATA
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_telco_customer_churn.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# TRAIN & HYPERPARAMETER SEARCH
# ===============================
best_acc = 0
best_model = None
best_pred = None
best_params = {}

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20]
}

for n in param_grid["n_estimators"]:
    for d in param_grid["max_depth"]:
        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=d,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_pred = y_pred
            best_params = {
                "n_estimators": n,
                "max_depth": d
            }

# ===============================
# SAVE MODEL & EVALUATION
# ===============================
os.makedirs("saved_models", exist_ok=True)
model_path = "saved_models/best_model_telco.pkl"
joblib.dump(best_model, model_path)

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
cm_path = "confusion_matrix_telco.png"
plt.savefig(cm_path)
plt.close()

# ===============================
# LOG TO MLFLOW (INI KUNCI)
# ===============================
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", best_params["n_estimators"])
    mlflow.log_param("max_depth", best_params["max_depth"])

    # Log metric
    mlflow.log_metric("accuracy", best_acc)

    # Log artifacts
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_artifact(cm_path, artifact_path="evaluation")

print("[SUCCESS] Training selesai & berhasil tercatat di MLflow")
print("Best Params:", best_params)
print("Best Accuracy:", best_acc)
