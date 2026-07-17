import requests

url = "https://nd0821-c3-starter-code-war4.onrender.com/predict"

payload = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 284582,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response:", response.text)
