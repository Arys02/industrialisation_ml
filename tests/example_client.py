import requests

from src.config import PORT

print("toto")
response = requests.post(f"http://127.0.0.1.0:{PORT}/hello")
print(response.text)
#response = requests.post(f"http://127.0.0.1:{PORT}/post_data", json=[{"year_week": 202001, "vegetable": "tomato", "sales": 100}])

print(response.json())


