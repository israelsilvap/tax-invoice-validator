from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH (se necessário)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Carregar o modelo treinado
model_path = "models/random_forest_model.pkl"
try:
    loaded_model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")

# Inicializa a API FastAPI
app = FastAPI()

# Modelo de dados esperado na requisição
class PredictionInput(BaseModel):
    iss_retention: float
    inss_tax_rate: float
    csll_tax_rate: float
    calculated_value: float
    cofins_tax_rate: float

@app.post("/predict/")
def get_prediction(data: PredictionInput):
    """
    Endpoint para prever a probabilidade de fraude.
    """
    try:
        # Converter entrada para numpy array (matriz 2D)
        X_input = np.array([[data.iss_retention, data.inss_tax_rate, data.csll_tax_rate,
                             data.calculated_value, data.cofins_tax_rate]])
        
        # Fazer a predição
        y_pred = loaded_model.predict(X_input)
        
        # Retornar resultado
        return {"fraud": int(y_pred[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a previsão: {e}")

@app.get("/test/")
def test():
    """
    Endpoint para testar se a API está rodando.
    """
    return {"message": "API está funcionando corretamente!"}
