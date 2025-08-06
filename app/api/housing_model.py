import os
import pickle
from logger import get_logger

# --- Constants ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "models", "model.pkl"))
INFERENCE_LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "inference_logs.txt"))
INFERENCE_COUNT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "count.txt"))
RETRAIN_THRESHOLD = 100

# --- Load model ---
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Logger ---
logger = get_logger("housing_model")

# --- Logging & Counter ---
os.makedirs(os.path.dirname(INFERENCE_LOG_FILE), exist_ok=True)

def update_inference_logs(features: dict, prediction: float):
    with open(INFERENCE_LOG_FILE, "a") as logf:
        logf.write(f"{features}, prediction={prediction}\n")

    count = 0
    if os.path.exists(INFERENCE_COUNT_FILE):
        with open(INFERENCE_COUNT_FILE, "r") as cf:
            count = int(cf.read().strip() or 0)

    count += 1
    with open(INFERENCE_COUNT_FILE, "w") as cf:
        cf.write(str(count))

    if count >= RETRAIN_THRESHOLD:
        logger.info("Retrain trigger threshold reached (count = 100)")

def predict_house_value(features: dict):
    try:
        required_keys = ['MedInc', 'HouseAge', 'AveBedrms', 'Latitude']
        if not all(key in features for key in required_keys):
            error = {"error": f"Missing required features: {required_keys}"}
            logger.warning(error)
            return error, 400

        X = [[features[key] for key in required_keys]]
        prediction = model.predict(X)[0]
        logger.info(f"Inference: features={features}, prediction={prediction:.2f}")
        update_inference_logs(features, prediction)
        return {"predicted_value": float(prediction)}, 200

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return {"error": str(e)}, 500
