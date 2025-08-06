import os
import pickle

# Get the absolute path to model.pkl relative to this file
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict_house_value(features: dict):
    try:
        required_keys = ['MedInc', 'HouseAge', 'AveBedrms', 'Latitude']
        if not all(key in features for key in required_keys):
            return {"error": f"Missing one or more required features: {required_keys}"}, 400

        X = [[features[key] for key in required_keys]]
        prediction = model.predict(X)[0]
        return {"predicted_value": float(prediction)}
    except Exception as e:
        return {"error": str(e)}, 500
