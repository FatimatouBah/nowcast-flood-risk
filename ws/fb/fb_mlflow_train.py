import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

data_path = 'src/fb/hubeau/'
files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'obstr' in f and '_H_' in f]

dfs = {}
for f in sorted(files):
    parts = f.replace('.csv', '').split('_')
    station = parts[2]
    key = f"{station}_H"
    df = pd.read_csv(data_path + f)
    df['date_obs'] = pd.to_datetime(df['date_obs'])
    df = df.sort_values('date_obs')
    dfs[key] = df

def assign_risk(df, col='resultat_obs'):
    q33 = df[col].quantile(0.33)
    q66 = df[col].quantile(0.66)
    df['risk_level'] = df[col].apply(lambda val: 1 if val <= q33 else (2 if val <= q66 else 3))
    return df

def add_features(df, col='resultat_obs'):
    df = df.sort_values('date_obs')
    df['hour'] = df['date_obs'].dt.hour
    df['dayofweek'] = df['date_obs'].dt.dayofweek
    df['month'] = df['date_obs'].dt.month
    df['rolling_mean_3h'] = df[col].rolling(window=18).mean()
    df['rolling_max_3h'] = df[col].rolling(window=18).max()
    df['rolling_std_3h'] = df[col].rolling(window=18).std()
    df['diff_1'] = df[col].diff(1)
    df['diff_6'] = df[col].diff(6)
    return df.dropna()

for key in list(dfs.keys()):
    dfs[key] = assign_risk(dfs[key])
    dfs[key] = add_features(dfs[key])

features = ['resultat_obs', 'hour', 'dayofweek', 'month', 'rolling_mean_3h', 'rolling_max_3h', 'rolling_std_3h', 'diff_1', 'diff_6']
target = 'risk_level'

df_model = pd.concat(list(dfs.values()))
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_experiment("nowcast-flood-risk")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.sklearn.log_model(model, "random_forest_model")
    print(f"F1: {f1:.3f}")
    print(classification_report(y_test, y_pred))
    print("Model logged to MLflow.")