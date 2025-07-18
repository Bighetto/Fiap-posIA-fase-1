from pydantic import BaseModel

class DadosPaciente(BaseModel):
    thal: int
    ca: int
    oldpeak: float
    exang: int
    cp: int
