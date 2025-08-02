import joblib
import numpy as np
from app.models.schema import PredictionInput, PredictionOutput
from app.utils.prometheus import prediction_counter, prediction_latency
from app.utils.logger import log_prediction

model = joblib.load("models/model.pkl")

@prediction_latency.time()
def make_prediction(input_data: PredictionInput) -> PredictionOutput:
    prediction_counter.inc()
    X = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    log_prediction(input_data.dict(), prediction)
    return PredictionOutput(prediction=prediction)
