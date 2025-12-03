# test_api.py
import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "subject": "Unable to login",
    "description": "Login fails with wrong password error"
}

response = requests.post(url, json=data)
print(response.json())
