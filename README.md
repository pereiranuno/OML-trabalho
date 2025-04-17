# Documentação do Projeto - Rumos Bank Lending Prediction


## Rumos Bank Going Live
Este projeto responde ao desafio proposto pelo Rumos Bank, que visa desenvolver uma solução de machine learning capaz de prever clientes que poderão falhar no cumprimento dos prazos de pagamento de crédito.

A prioridade do banco é garantir que a transição dos resultados exploratórios para produção é feita de forma eficiente e automatizada, evitando demoras como em experiências anteriores.


> [!IMPORTANT]
> Esta secção contém observações relevantes para garantir a correta execução do projeto.
> A imagem Docker do serviço encontra-se publicada de forma pública no GitHub Container Registry (GHCR).
> 🔗 Imagem: `ghcr.io/pereiranuno/bank_lending_prediction_service:latest`
> O serviço não inclui o modelo diretamente na imagem, pois o mesmo é carregado dinamicamente do **MLflow Tracking Server**, a partir do Model Registry. A versão utilizada é a `champion` do modelo `random_forest`
> Uma instância do MLflow é levantado via `docker-compose` e pode ser acedido localmente em http://localhost:5000
> O ficheiro `conda.yaml` define todas as dependências necessárias para reproduzir o ambiente localmente.
> Pode ser usado com:
 ```bash 
    conda env create -f conda.yaml
    conda activate rumos_bank_lending_
```



## Dependencias
- Python 3.10
- FastAPI
- Scikit-learn
- MLflow
- Conda
- Docker
- Pytest
- uvicorn
- ipykernel
- numpy
- pandas
- pydantic
- cloudpickle
- matplotlib
- requests



## Estrutura Projecto
```plaintext
OML-trabalho/
├── .github/
│   └── workflows/
│       └── cicd.yaml
├── config/
│   └── app.json
├── data/
│   └── lending_data.csv
├── notebooks/
│   └── mlflow/
│       ├── mlflow_model_read.ipynby
│       ├── mlflow_model_reg.ipynby
│   └── rumos_bank_lending_prediction.ipynb
├── src/
│   └── app/
│       ├── main.py
├── tests/
│   ├── test_main.py
│   └── test_model.py
├── conda.yaml
├── docker-compose.yml
├── Dockerfile.Service
└── README.md
```

### Definição Esrutura

- `src/app/main.py` - Código permite expor modelo ML como um serviço utilizando API FastAPI
- `config/app.json` - Configuração do serviço e modelo champion
- `notebooks/rumos_bank_lending_prediction.ipynb` - Notebook exploratório utilizado para otimizar os modelos existentes
- `notebooks/mlflow/mlflow_model_reg.ipynb` - Notebook que regista os modelos e toda e artefactos e experiências no model registry
- `notebooks/mlflow/mlflow_model_read.ipynb` - Notebook que testa a leitura do modelo champion e executa uma predição para um conjunto de inputs aleatórios.
- `conda.yaml` - Ambiente conda com a definição das dependências do projecto
- `Dockerfile.Service` - Dockerfile do serviço do modelo
- `docker-compose.yml` - Orquestração de serviços (MLflow + Modelo como Serviço API)
- `tests/` - Testes unitários da serviço e testes ao  modelo
- `.github/workflows/pipeline.yml` - Pipeline de CI/CD

---


## Reproduzibilidade

Para correr o projeto localmente:

```bash
git clone https://github.com/pereiranuno/OML-trabalho.git
cd OML-trabalho
```
Subir os serviços (MLflow + API)
````docker compose up -d```

Aceder à API: http://localhost:5002/docs
Aceder ao MLflow: http://localhost:5000

Para testar localmente criar o ambiente através
```bash
conda env create -f conda.yaml
conda activate rumos_bank_lending_
pytest
```


## ML Model Registry

Os modelos são treinados e registados com o MLflow, onde a versão com melhor desempenho é promovida para champion.

O serviço consome o modelo diretamente do MLflow Tracking Server, lendo a configuração do modelo em config/app.json a ser utilizado como serviço.

```python
mlflow.set_tracking_uri("http://mlflow-tracking-server:5000")
mlflow.pyfunc.load_model("models:/random_forest@champion")
```

![alt text](utils/pics/experiments.png)

![alt text](utils/pics/compare1.png)

![alt text](utils/pics/compare2.png)

![alt text](utils/pics/compare3.png)



## Modelo como um serviço

- O modelo treinado encontra-se exposto através de uma **API REST criada com FastAPI**.
- O endpoint principal de previsão está acessível em: `POST http://localhost:5002/predict_bank_lending`
- A comunicação com a API pode ser feita via Postman ou curl.

#### Exemplo de comunicação:
**Request:**
```http
POST http://localhost:5002/predict_bank_lending
Content-Type: application/json

{
  "LIMIT_BAL": 2000,
  "SEX": 2,
  "EDUCATION": 1,
  "MARRIAGE": 2,
  "AGE": 26,
  "PAY_0": 1,
  "PAY_2": 2,
  "PAY_3": 2,
  "PAY_4": 2,
  "PAY_5": 2,
  "PAY_6": 2,
  "BILL_AMT1": 1001,
  "BILL_AMT2": 1200,
  "BILL_AMT3": 1300,
  "BILL_AMT4": 1249,
  "BILL_AMT5": 1000,
  "BILL_AMT6": 1000,
  "PAY_AMT1": 1000,
  "PAY_AMT2": 1000,
  "PAY_AMT3": 1000,
  "PAY_AMT4": 1000,
  "PAY_AMT5": 1000,
  "PAY_AMT6": 1000
}
```
**Response:**
```http
{
  "prediction": 0
}
```

## CI/CD

O projeto inclui uma pipeline GitHub Actions que:

1. Clona o repositório
2. Constrói os serviços via Docker Compose
3. Cria o ambiente Conda e executa os testes com `pytest`
4. Faz login no GHCR
5. Publica a imagem para o container registry

O acesso à imagem é garantido através do `GITHUB_TOKEN`, pois o repositório está devidamente ligado ao package no GHCR com as respetivas permissões.

---


## Autor
Nuno Pereira
github.com/pereiranuno
pereiranuno@gmail.com





