import json
import pytest
import requests

with open('./config/app.json') as f:
    config = json.load(f)

def test_has_diabetes_prediction():
    """
    Test for the /predict_bank_lending endpoint with valid input data.
    It should return a prediction in the response.
    """
    response = requests.post(f'http://localhost:{config["service_port"]}/predict_bank_lending', json={

    'LIMIT_BAL': 2000,
    'SEX': 2,
    'EDUCATION': 1,
    'MARRIAGE': 2,
    'AGE': 26,
    'PAY_0': 1,
    'PAY_2': 2,
    'PAY_3': 2,
    'PAY_4': 2,
    'PAY_5': 2,
    'PAY_6': 2,
    'BILL_AMT1': 1001,
    'BILL_AMT2': 1200,
    'BILL_AMT3': 1300,
    'BILL_AMT4': 1249,
    'BILL_AMT5': 1000,
    'BILL_AMT6': 1000,
    'PAY_AMT1': 1000,
    'PAY_AMT2': 1000,
    'PAY_AMT3': 1000,
    'PAY_AMT4': 1000,
    'PAY_AMT5': 1000,
    'PAY_AMT6': 1000
    
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))
    assert response.json()["prediction"] == 0