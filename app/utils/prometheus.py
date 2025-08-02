from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter("predictions_total", "Total number of prediction requests")
prediction_latency = Histogram("prediction_latency_seconds", "Time spent on prediction")

def get_metrics():
    return generate_latest()
