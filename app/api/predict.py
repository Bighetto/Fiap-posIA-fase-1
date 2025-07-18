import joblib
import numpy as np
from app.api.schemas import DadosPaciente

model = joblib.load("app/model/modelo.pkl")
scaler = joblib.load("app/model/scaler.pkl")

def prever_doenca(data: DadosPaciente):
    entrada = np.array([[data.thal, data.ca, data.oldpeak, data.exang, data.cp]])
    entrada_scaled = scaler.transform(entrada)
    pred = model.predict(entrada_scaled)[0]
    return "Doença detectada" if pred == 1 else "Sem doença detectada"
