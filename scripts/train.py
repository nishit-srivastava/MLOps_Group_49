import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load dataset
df = pd.read_csv("data/processed/cleaned.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and log models
models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor()
}

best_rmse = float("inf")
best_model_name = None
best_model = None

with mlflow.start_run():
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mlflow.log_metric(f"{name}_rmse", rmse)
        mlflow.sklearn.log_model(model, f"{name}_model")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

    mlflow.set_tag("best_model", best_model_name)
    joblib.dump(best_model, "models/model.pkl")
    print(f"Best model saved: {best_model_name} with RMSE: {best_rmse}")
