import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import mlflow
import mlflow.sklearn

# ===============================
# PATH SETUP (PREPROCESSED DATA)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "..", "..", "data", "preprocessed_telco_customer_churn.csv"
)

print(f"[INFO] Load dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ===============================
# FEATURE & TARGET
# ===============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# HYPERPARAMETER GRID
# ===============================
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20]
}

best_acc = 0
best_model = None
best_params = {}
best_y_pred = None

print("[INFO] Mulai Hyperparameter Tuning RandomForest...")

# ===============================
# PARENT MLFLOW RUN
# ===============================
with mlflow.start_run(run_name="RF_Tuning_Telco_Parent"):

    for n in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:

            run_name = f"n_estimators={n}_max_depth={depth}"

            with mlflow.start_run(run_name=run_name, nested=True):

                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    random_state=42
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # ===============================
                # LOG PARAM & METRIC
                # ===============================
                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", depth)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)

                print(f"[RUN] {run_name} | Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {
                        "n_estimators": n,
                        "max_depth": depth
                    }
                    best_y_pred = y_pred

    # ===============================
    # LOG BEST RESULT
    # ===============================
    print("\n[RESULT] BEST MODEL")
    print("Best Params :", best_params)
    print(f"Best Accuracy : {best_acc:.4f}")

    mlflow.log_param("best_n_estimators", best_params["n_estimators"])
    mlflow.log_param("best_max_depth", best_params["max_depth"])
    mlflow.log_metric("best_accuracy", best_acc)

    # ===============================
    # SAVE BEST MODEL
    # ===============================
    mlflow.sklearn.log_model(best_model, "best_model_telco")

    # ===============================
    # ARTIFACT: CONFUSION MATRIX
    # ===============================
    cm = confusion_matrix(y_test, best_y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Best RF Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_file = "confusion_matrix_telco.png"
    plt.savefig(cm_file)
    plt.close()
    mlflow.log_artifact(cm_file)

    # ===============================
    # ARTIFACT: FEATURE IMPORTANCE
    # ===============================
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - Best RF Model")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
    plt.tight_layout()

    fi_file = "feature_importance_telco.png"
    plt.savefig(fi_file)
    plt.close()
    mlflow.log_artifact(fi_file)

    # Cleanup local files
    os.remove(cm_file)
    os.remove(fi_file)

print("[SUCCESS] Hyperparameter tuning Telco Churn selesai.")
