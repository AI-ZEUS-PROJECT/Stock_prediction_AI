import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "ticker": "AAPL",
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
    "time_step": 100
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())