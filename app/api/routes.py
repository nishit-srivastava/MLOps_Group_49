from fastapi import APIRouter
from app.models.schema import PredictionInput, PredictionOutput
from app.services.predictor import make_prediction

router = APIRouter()

@router.post("/", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    return make_prediction(input_data)
