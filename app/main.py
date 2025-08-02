from fastapi import FastAPI, Response
from app.api.routes import router
from prometheus_client import CONTENT_TYPE_LATEST
from app.utils.prometheus import get_metrics

app = FastAPI(title="California Housing Predictor")
app.include_router(router, prefix="/predict")

@app.get("/metrics")
def metrics():
    return Response(get_metrics(), media_type=CONTENT_TYPE_LATEST)
