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
    iss_retention: str  # Vai ser "Aplicável" ou "Não Aplicável"
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
        # Tratamento para o campo iss_retention (convertendo para 1 e 0)
        if data.iss_retention == "Aplicável":
            iss_retention_value = 1
        elif data.iss_retention == "Não Aplicável":
            iss_retention_value = 0
        else:
            raise HTTPException(status_code=400, detail="Valor inválido para iss_retention. Use 'Aplicável' ou 'Não Aplicável'.")

        # Garantir que as taxas estão entre 0 e 100
        iss_tax_rate = min(max(data.inss_tax_rate, 0), 100)
        inss_tax_rate = min(max(data.inss_tax_rate, 0), 100)
        csll_tax_rate = min(max(data.csll_tax_rate, 0), 100)
        cofins_tax_rate = min(max(data.cofins_tax_rate, 0), 100)

        # Converter entrada para numpy array (matriz 2D)
        X_input = np.array([[iss_retention_value, iss_tax_rate, inss_tax_rate,
                             csll_tax_rate, cofins_tax_rate]])

        # Fazer a predição
        y_pred = loaded_model.predict(X_input)

        # Retornar resultado
        return {"prediction": int(y_pred[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a previsão: {e}")

@app.get("/test/")
def test():
    """
    Endpoint para testar se a API está rodando.
    """
    return {"message": "API está funcionando corretamente!"}
