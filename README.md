
#Steps to run API 

cd app
cd api
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
 python app.py

curl --location 'http://localhost:5000/predict' \
--header 'Content-Type: application/json' \
--data '{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveBedrms": 1.02,
  "Latitude": 37.88
}

//build image locally or get from docker hub
docker build -t mlops-flask-api:latest -f ./docker/Dockerfile ./app/api

use docker compose to run all
//docker compose up
