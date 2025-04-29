import requests

comment = {'sample_index': 1}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=comment)
print(response.json())