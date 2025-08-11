# MLOps Flask API with Prometheus & Grafana Monitoring

This project sets up a **Flask API** with **Prometheus** metrics and **Grafana** dashboards using Docker Compose.  
The API provides a machine learning model prediction endpoint and exposes metrics for monitoring.

---

## 📂 Project Structure

.
├── app
│ └── api # Flask API source code
├── prometheus
│ └── prometheus.yml # Prometheus configuration
├── docker-compose.yml
└── README.md


---

## 📦 Services Overview

### 1. **API (`mlops-flask-api`)**
- Flask API exposing `/predict` endpoint
- Exposes Prometheus metrics
- Runs on port **5000**
- Uses volume mount for live code reload

### 2. **Prometheus**
- Scrapes metrics from the API
- Configurable via `prometheus/prometheus.yml`
- Runs on port **9090**

### 3. **Grafana**
- Provides dashboards for metrics visualization
- Runs on port **3000**
- Default admin password: `admin`
- Persists data in `grafana-storage` volume

---

## 🔹 Docker Compose Network & Volumes

- **Network:** `monitor-net` — Shared network for all services
- **Volume:** `grafana-storage` — Persistent Grafana data






curl --location 'http://localhost:5000/predict' \
--header 'Content-Type: application/json' \
--data '{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveBedrms": 1.02,
  "Latitude": 37.88
}'

# Build and start services
docker-compose up --build

API → http://localhost:5000

Prometheus → http://localhost:9090

Grafana → http://localhost:3000

Username: admin
Password: admin

Add Prometheus as a data source:

URL: http://prometheus:9090


Stopping Services

docker-compose down

## 🚀 Steps to Run API Without Docker
If you want to run the API manually instead of Docker:

```bash
# Navigate to API folder
cd app
cd api

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the API
python app.py
