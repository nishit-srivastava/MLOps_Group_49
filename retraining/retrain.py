
import numpy as np
import pandas as pd
import os 
import sqlite3
from feast import FeatureStore

from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

import subprocess
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import joblib
import glob
import os 
import pyarrow as pa
import pyarrow.parquet as pq


#read new data 

username = os.getenv("GIT_USERNAME")
token = os.getenv("GIT_TOKEN")

files = glob.glob("../new_data/*.csv")
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list)
df = df.drop_duplicates()

df.to_csv('../data/housing.csv', index=False)



df = df.drop(['AveRooms', 'Longitude','Population','AveOccup'], axis=1)


df['house_id'] = range(1, len(df) + 1)

# Add the current timestamp (same for all rows)
df['event_timestamp'] = datetime.now()

parquet_path = "../feature_store/housing_feature_repo/feature_repo/data/housing_features.parquet"
table = pa.Table.from_pandas(df)
pq.write_table(table, parquet_path)

store = FeatureStore("../feature_store/housing_feature_repo/feature_repo")
store.materialize_incremental(end_date=datetime.utcnow())


entity_df = pd.read_parquet(parquet_path)[[
    "event_timestamp",
    "house_id",
    "MedHouseVal"
]]

# 3. Retrieve historical features to build the training dataset
print("Retrieving training data from Feast...")
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "location_features:MedInc",
        "location_features:HouseAge",
        "location_features:AveBedrms",
        "location_features:Latitude",
    ],
).to_df()

print("\n--- Training Data ---")
print(training_data.head())

# 4. Train a model
print("\n--- Training Model ---")

# Define features (X) and target (y)
X = training_data[['MedInc', 'HouseAge', 'AveBedrms', 'Latitude']]
y = training_data['MedHouseVal']

# Split data for training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Replace these with your actual training data
# X_train, X_test, y_train, y_test must be defined before running this script

# Set MLflow experiment
mlflow.set_experiment("Retraining")

best_overall_model = None
best_overall_r2 = float('-inf')
best_model_name = ""
best_run_id = ""
best_artifact_path = ""


# --- Decision Tree with Grid Search ---
with mlflow.start_run(run_name="Tuned_Decision_Tree") as run_dt:
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_dt_model = grid_search.best_estimator_
    dt_preds = best_dt_model.predict(X_test)

    mse_dt = float(mean_squared_error(y_test, dt_preds))
    r2_dt = float(r2_score(y_test, dt_preds))

    best_params = {
        k: (str(v) if v is None else v)
        for k, v in grid_search.best_params_.items()
    }

    mlflow.log_param("model_type", "DecisionTreeRegressor")
    mlflow.log_params(best_params)
    mlflow.log_metric("mse", mse_dt)
    mlflow.log_metric("r2", r2_dt)

    artifact_path_dt = "decision_tree_model"
    mlflow.sklearn.log_model(
        sk_model=best_dt_model,
        artifact_path=artifact_path_dt,
        registered_model_name="DecisionTreeRegressorModel"
    )

    print("\nTuned Decision Tree Regressor:")
    print(f"  Best Parameters: {best_params}")
    print(f"  MSE: {mse_dt:.4f}")
    print(f"  R²: {r2_dt:.4f}")

    if r2_dt > best_overall_r2:
        best_overall_model = best_dt_model
        best_overall_r2 = r2_dt
        best_model_name = "DecisionTreeRegressorModel"
        best_run_id = run_dt.info.run_id
        best_artifact_path = artifact_path_dt

# --- Export metrics safely to YAML ---
client = MlflowClient()

mse_history = client.get_metric_history(best_run_id, "mse")
r2_history = client.get_metric_history(best_run_id, "r2")

# Convert to primitive dictionaries
mse_safe = [{
    "key": m.key,
    "value": m.value,
    "step": m.step,
    "timestamp": m.timestamp,
    "run_id": m.run_id,
    "model_id": getattr(m, "model_id", None)
} for m in mse_history]

r2_safe = [{
    "key": m.key,
    "value": m.value,
    "step": m.step,
    "timestamp": m.timestamp,
    "run_id": m.run_id,
    "model_id": getattr(m, "model_id", None)
} for m in r2_history]

# Save to YAML
with open("best_model_metrics.yaml", "w") as f:
    yaml.dump({
        "best_model_name": best_model_name,
        "best_run_id": best_run_id,
        "mse_history": mse_safe,
        "r2_history": r2_safe
    }, f)


# --- Final Output ---
print(f"\n✅ Best Model: {best_model_name} with R² = {best_overall_r2:.4f}")
print(f"   Run ID: {best_run_id}")
print("   Metrics saved to: best_model_metrics.yaml")




# --- Register the best model ---
model_uri = f"runs:/{best_run_id}/{best_artifact_path}"
registration_result = mlflow.register_model(
    model_uri=model_uri,
    name="HousingPricePredication"
)

import pickle

import os
import glob
import re
import subprocess
import shutil

models_dir = os.path.join("../app/api/", "models")

# Get all files matching model_v*.pkl
model_files = glob.glob(f"{models_dir}/model_v*.pkl")

if not model_files:
    # No versioned files found, start at v1
    new_model_path = os.path.join(models_dir, "model_v1.pkl")
else:
    # Find latest model by version number (not time)
    def extract_version(filename):
        match = re.search(r"model_v(\d+)\.pkl", os.path.basename(filename))
        return int(match.group(1)) if match else -1

    latest_model = max(model_files, key=extract_version)
    latest_version = extract_version(latest_model)
    new_version = latest_version + 1
    new_model_path = os.path.join(models_dir, f"model_v{new_version}.pkl")

    # Delete the older version
    os.remove(latest_model)
    print(f"Deleted old model: {latest_model}")

# Simulate creating the new model file (replace with your actual model saving code)
with open(new_model_path, "wb") as f:
     pickle.dump(best_dt_model, f)  # Placeholder for actual model bytes
print(f"Created new model: {new_model_path}")

# Add and commit to git
repo_path = "../../MLOps_Group_49"
subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
subprocess.run(["git", "commit", "-m", f"chore: add model_v{new_version}.pkl"], cwd=repo_path, check=True)
#subprocess.run(["git", "pull", "--rebase"])
subprocess.run(["git", "push"], cwd=repo_path, check=True)



