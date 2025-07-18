from fastapi import FastAPI
from app.api.schemas import DadosPaciente
from app.api.predict import prever_doenca

app = FastAPI(
    title="API de Doença Cardíaca",
    description="Regressão logística para prever doenças com base em dados clínicos",
    version="1.0"
)

@app.get("/")
def read_root():
    return {"mensagem": "API funcionando!"}

@app.post("/predict")
def predict(data: DadosPaciente):
    resultado = prever_doenca(data)
    return {"resultado": resultado}
