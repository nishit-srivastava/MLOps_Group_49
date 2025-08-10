from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, ValidationError
from housing_model import predict_house_value
from logger import get_logger
from flasgger import Swagger

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)  # Initialize Swagger#
logger = get_logger("api")

# Prometheus Metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
REQUEST_LATENCY = Summary('inference_request_latency_seconds', 'Request latency in seconds')

# Input validation schema
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveBedrms: float
    Latitude: float

@app.route('/')
def home():
    """
    Home endpoint
    ---
    responses:
      200:
        description: API status
    """
    return jsonify({"message": "Inference API is running"}), 200

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time()
def predict():
    """
    Predict house value
    ---
    tags:
      - Inference
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            MedInc:
              type: number
              example: 3.5
            HouseAge:
              type: number
              example: 15
            AveBedrms:
              type: number
              example: 1.0
            Latitude:
              type: number
              example: 34.05
    responses:
      200:
        description: Predicted value
      400:
        description: Validation error
      500:
        description: Internal server error
    """
    REQUEST_COUNT.inc()
    try:
        input_json = request.get_json()
        input_data = HousingInput(**input_json)
        result, status = predict_house_value(input_data.dict())
        return jsonify(result), status
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve.errors()}")
        return jsonify({"error": ve.errors()}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    """
    Prometheus metrics
    ---
    responses:
      200:
        description: Prometheus metrics output
    """
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
