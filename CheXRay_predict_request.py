import requests
# This code sends request for prediction of inference files that are stored locally. It sends the approach to be used for prediction.

url = 'http://localhost:5000/'
r = requests.post(url,json={"approach":"u-Ones"})
print(r.json())