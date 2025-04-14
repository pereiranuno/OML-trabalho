# main.py

import fastapi
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from pydantic import BaseModel, conint, confloat
import pandas as pd
import json
import uvicorn

# Carregar configuração da aplicação
with open('./config/app.json') as f:
    config = json.load(f)

# Modelo de input com todos os campos esperados pelo modelo ML
class Request(BaseModel):

    """
    LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
    SEX: Gender (1=male, 2=female)
    EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
    MARRIAGE: Marital status (1=married, 2=single, 3=others)
    AGE: Age in years
    PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
    PAY_2: Repayment status in August, 2005 (scale same as above)
    PAY_3: Repayment status in July, 2005 (scale same as above)
    PAY_4: Repayment status in June, 2005 (scale same as above)
    PAY_5: Repayment status in May, 2005 (scale same as above)
    PAY_6: Repayment status in April, 2005 (scale same as above)
    BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
    BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
    BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
    BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
    BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
    BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
    PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
    PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
    PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
    PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
    PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
    PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)

    """


    LIMIT_BAL: float = 2000.0
    SEX: int = 2
    EDUCATION: int = 1
    MARRIAGE: int = 2
    AGE: int = 26
    PAY_0: int = 1
    PAY_2: int = 2
    PAY_3: int = 2
    PAY_4: int = 2
    PAY_5: int = 2
    PAY_6: int = 2
    BILL_AMT1: float = 1001.0
    BILL_AMT2: float = 1200.0
    BILL_AMT3: float = 1300.0
    BILL_AMT4: float = 1249.0
    BILL_AMT5: float = 1000.0
    BILL_AMT6: float = 1000.0
    PAY_AMT1: float = 1000.0
    PAY_AMT2: float = 1000.0
    PAY_AMT3: float = 1000.0
    PAY_AMT4: float = 1000.0
    PAY_AMT5: float = 1000.0
    PAY_AMT6: float = 1000.0

# Criar aplicação FastAPI
app = fastapi.FastAPI()

# Permitir CORS para facilitar testes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelo com base na configuração
@app.on_event("startup")
async def startup_event():

    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Load the registered model specified in the configuration
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri = model_uri)
    
    print(f"Loaded model {model_uri}")

# Endpoint para previsão
@app.post("/predict_bank_lending")
async def predict(input: Request):
    """
    Prediction endpoint that processes input data and returns a model prediction.

    Parameters:
        input (Request): Request body containing input values for the model.

    Returns:
        dict: A dictionary with the model prediction under the key "prediction".
    """

    # Build a DataFrame from the request data
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.model_dump().items()})
    print(input_df.sample(1))

    # Predict using the model and retrieve the first item in the prediction list
    prediction = app.model.predict(input_df)

    # Return the prediction result as a JSON response
   
    return {"prediction": prediction.tolist()[0]}

# Executar aplicação
uvicorn.run(app=app, port=config["service_port"], host="0.0.0.0")
