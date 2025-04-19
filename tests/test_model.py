import json
import pytest
import pandas as pd
import mlflow


@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}")
    model_name = config["model_name"]
    model_version = config["model_version"]
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )


def test_model_out(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{

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

    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0


def test_model_dir(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{

    'LIMIT_BAL': 30000,
    'SEX': 2,
    'EDUCATION': 2,
    'MARRIAGE': 1,
    'AGE': 31,
    'PAY_0': 3,
    'PAY_2': 3,
    'PAY_3': 2,
    'PAY_4': 2,
    'PAY_5': 2,
    'PAY_6': 2,
    'BILL_AMT1': 26008,
    'BILL_AMT2': 25294,
    'BILL_AMT3': 26928,
    'BILL_AMT4': 0,
    'BILL_AMT5': 1600,
    'BILL_AMT6': 1200,
    'PAY_AMT1': 0,
    'PAY_AMT2': 2200,
    'PAY_AMT3': 0,
    'PAY_AMT4': 0,
    'PAY_AMT5': 0,
    'PAY_AMT6': 0

    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0


def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{

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


    
    
    }])
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )