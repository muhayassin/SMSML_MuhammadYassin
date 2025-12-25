import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "..", "..", "data", "preprocessed_telco_customer_churn.csv"
)

# ===============================
# MLFLOW AUTOLOG
# ===============================
mlflow.set_experiment("Telco-Churn-Experiment")
mlflow.autolog()

# ===============================
# LOAD DATA (PREPROCESSED)
# ===============================
df = pd.read_csv(DATA_PATH)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# MODEL
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Test Accuracy:", acc)
