from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route("/retrain", methods=["POST"])
def retrain():
    print("Retrain webhook received!")
    subprocess.Popen(["python", "retrain.py"])
    return "Retraining started", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
