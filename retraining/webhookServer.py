from flask import Flask, request
import subprocess
import threading

app = Flask(__name__)
lock = threading.Lock()

@app.route("/retrain", methods=["POST"])
def retrain():
    if lock.locked():
        return "Retraining already in progress. Try again later.", 429  # Too Many Requests

    with lock:  # Prevents other retrain calls until this finishes
        print("Retrain webhook received!")
        result = subprocess.run(["python", "retrain.py"])
        
        if result.returncode == 0:
            return "Retraining completed successfully", 200
        else:
            return f"Retraining failed with code {result.returncode}", 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
