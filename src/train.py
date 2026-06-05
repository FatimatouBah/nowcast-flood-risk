import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

data_path = "src/fb/hubeau/"
files = [f for f in os.listdir(data_path) if f.endswith(".csv") and "obstr" in f]

dfs = []
for f in files:
    df = pd.read_csv(os.path.join(data_path, f))
    df["date_obs"] = pd.to_datetime(df["date_obs"])
    df = df.sort_values("date_obs")
    dfs.append(df)

df = pd.concat(dfs).sort_values("date_obs")
df["risk_level"] = pd.qcut(df["resultat_obs"], 3, labels=[1,2,3])
df["risk_level"] = df["risk_level"].astype(int)
df["hour"] = df["date_obs"].dt.hour
df["month"] = df["date_obs"].dt.month
df = df.dropna()

features = ["resultat_obs", "hour", "month"]
target = "risk_level"

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

mlflow.set_experiment("flood-risk")

experiments = [
    ("RandomForest_50", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("RandomForest_100", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("RandomForest_200", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("XGBoost", xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')),
]

for run_name, model in experiments:
    with mlflow.start_run(run_name=run_name):
        if "XGBoost" in run_name:
            model.fit(X_train, y_train - 1)
            pred = model.predict(X_test) + 1
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        f1 = f1_score(y_test, pred, average="weighted")
        mlflow.log_param("model", run_name)
        mlflow.log_param("n_estimators", getattr(model, 'n_estimators', 0))
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, "model")
        print(f"{run_name} — F1: {f1:.3f}")