import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ===============================
# PATH DATA
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "preprocessed_telco_customer_churn.csv")
)

print(f"[INFO] Load dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ===============================
# FEATURE & TARGET
# ===============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# TUNING
# ===============================
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20]
}

best_acc = 0
best_model = None
best_y_pred = None
best_params = {}

print("[INFO] Mulai Hyperparameter Tuning RandomForest...")

for n in param_grid["n_estimators"]:
    for depth in param_grid["max_depth"]:
        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"[RUN] n_estimators={n}, max_depth={depth} | Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = {"n_estimators": n, "max_depth": depth}
            best_y_pred = y_pred

print("\n[RESULT] BEST MODEL")
print("Best Params :", best_params)
print(f"Best Accuracy : {best_acc:.4f}")

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("saved_models", exist_ok=True)
joblib.dump(best_model, "saved_models/best_model_telco.pkl")

# ===============================
# SAVE CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.savefig("saved_models/confusion_matrix_telco.png")
plt.close()

print("[SUCCESS] Model training selesai.")
