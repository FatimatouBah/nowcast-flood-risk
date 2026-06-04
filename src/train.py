import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

data_path = "src/fb/hubeau/"

files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

dfs = []

for f in files:
    df = pd.read_csv(os.path.join(data_path, f))
    df["date_obs"] = pd.to_datetime(df["date_obs"])
    df = df.sort_values("date_obs")
    dfs.append(df)

df = pd.concat(dfs).sort_values("date_obs")

df["risk_level"] = pd.qcut(df["resultat_obs"], 3, labels=[1,2,3])

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

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    f1 = f1_score(y_test, pred, average="weighted")

    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(model, "model")

    print("F1:", f1)