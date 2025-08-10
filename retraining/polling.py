import requests
import time

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
PROMQL = 'inference_requests_total'
RETRAIN_HOOK_URL = 'http://localhost:5001/retrain'

POLL_INTERVAL = 10  # seconds
THRESHOLD_STEP = 10  # retrain every 100 requests

last_triggered_threshold = 0

def get_request_count():
    try:
        response = requests.get(PROMETHEUS_URL, params={'query': PROMQL})
        data = response.json()

        # Extract the value of the counter
        if data['status'] == 'success' and data['data']['result']:
            value = float(data['data']['result'][0]['value'][1])
            return int(value)
    except Exception as e:
        print(f"Failed to query Prometheus: {e}")
    return None

def trigger_retrain(current_value):
    try:
        print(f"Triggering retraining at request count: {current_value}")
        requests.post(RETRAIN_HOOK_URL)
    except Exception as e:
        print(f"Failed to trigger retraining: {e}")

def main():
    global last_triggered_threshold

    print("Starting Prometheus polling loop...")
    while True:
        count = get_request_count()
        if count is not None:
            next_threshold = ((count // THRESHOLD_STEP) * THRESHOLD_STEP)
            print(count)
            print(next_threshold) 
            if next_threshold > last_triggered_threshold and count % THRESHOLD_STEP == 0:
                trigger_retrain(count)
                last_triggered_threshold = next_threshold
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
