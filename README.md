# Documentação do Projeto - Rumos Bank Lending Prediction


## Rumos Bank Going Live
Este projeto responde ao desafio proposto pelo Rumos Bank, que visa desenvolver uma solução de machine learning capaz de prever clientes que poderão falhar no cumprimento dos prazos de pagamento de crédito.

A prioridade do banco é garantir que a transição dos resultados exploratórios para produção é feita de forma eficiente e automatizada, evitando demoras como em experiências anteriores.


> [!IMPORTANT]
> Esta secção contém observações relevantes para garantir a correta execução do projeto.
> A imagem Docker do serviço encontra-se publicada de forma pública no GitHub Container Registry (GHCR).
> 🔗 Imagem: `ghcr.io/pereiranuno/bank_lending_prediction_service:latest`
> O serviço não inclui o modelo diretamente na imagem, carregado dinamicamente do **MLflow Tracking Server**, a partir do Model Registry. A versão utilizada é a `champion` do modelo `random_forest`
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
- GitHub Actions
- GHCR (GitHub Container Registry)
- Pytest

## Estrutura Projecto
```plaintext
OML-trabalho/
├── .github/
│   └── workflows/
│       └── cicd.yaml
├── config/
│   └── app.json
├── notebooks/
│   └── rumos_bank_lending_prediction.ipynb
├── src/
│   └── app/
│       ├── main.py
│       ├── model.py
│       └── utils.py
├── tests/
│   ├── test_main.py
│   └── test_model.py
├── conda.yaml
├── docker-compose.yml
├── Dockerfile.Service
└── README.md
```

### Definição Esrutura

- `src/` - Código da API FastAPI
- `config/app.json` - Configuração do serviço e modelo
- `notebooks/rumos_bank_lending_prediction.ipynb` - Notebook exploratório
- `notebooks/mlflow/mlflow_model_reg.ipynb` - Notebook que regista os modelos e toda e artefactos e experiências no model registry
- `notebooks/mlflow/mlflow_model_read.ipynb` - Notebook que testa a leitura do modelo champion e executa uma predição para um conjunto de inputs aleatórios.
- `conda.yaml` - Ambiente conda com dependências
- `Dockerfile.Service` - Dockerfile do serviço
- `docker-compose.yml` - Orquestração de serviços (MLflow + Modelo como Serviço API)
- `tests/` - Testes unitários da API e modelo
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

O serviço consome o modelo diretamente do MLflow Tracking Server, lendo a configuração do modelo em config/app.json.

```bash

mlflow.set_tracking_uri("http://mlflow-tracking-server:5000")
mlflow.pyfunc.load_model("models:/random_forest@champion")
```

![alt text](utils/pics/experiments.png)

![alt text](utils/pics/compare1.png)

![alt text](utils/pics/compare2.png)

![alt text](utils/pics/compare3.png)


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


asa




