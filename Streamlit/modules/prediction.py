import streamlit as st
import pandas as pd
import requests

API_URL = "http://18.119.19.95/predict"  # Altere para o endpoint correto da sua API

def prediction():
    """
    Aba de Previsão de Fraude.

    Faz uma requisição à API para prever a probabilidade de fraude.
    """

    st.markdown("<h3 style='text-align: center;'>Detecção de Notas Fiscais com Impostos Incorretos</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>Utilize a API para classificar notas fiscais</h6>", unsafe_allow_html=True)

    # Formulário para inserção dos dados
    with st.form("fraud_prediction_form"):
        # Alterando para "Aplicável" e "Não Aplicável" no lugar de True e False
        iss_retention = st.selectbox("Retenção de ISS", options=["Aplicável", "Não Aplicável"])
        iss_retention = 1 if iss_retention == "Aplicável" else 0

        # Usando o slider para as taxas (garantindo que os valores sejam float)
        iss_tax_rate = st.slider("Taxa de ISS aplicada sobre o valor da nota fiscal", 
                                 min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        
        inss_tax_rate = st.slider("Taxa de INSS aplicada, se for o caso", 
                                  min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        
        csll_tax_rate = st.slider("Taxa de CSLL aplicada, se for o caso", 
                                  min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        
        cofins_tax_rate = st.slider("Taxa de COFINS (Contribuição para o Financiamento da Seguridade Social)", 
                                    min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        
        # O valor total calculado pode ser ajustado conforme necessário
        calculated_value = st.number_input("Valor total calculado da nota fiscal, incluindo impostos", 
                                          value=0.0, format="%.2f")
        
        submit_button = st.form_submit_button("Fazer Previsão")

    if submit_button:
        # Criar o JSON com os dados do usuário
        payload = {
            "iss_retention": iss_retention,
            "inss_tax_rate": inss_tax_rate,
            "csll_tax_rate": csll_tax_rate,
            "calculated_value": calculated_value,
            "cofins_tax_rate": cofins_tax_rate
        }

        # Fazer a requisição à API
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Levanta erro para status HTTP 4xx ou 5xx
            result = response.json()  # Converte resposta para JSON

            # Exibir o resultado da previsão
            st.subheader("Resultado da Previsão")
            if result["prediction"] == 1:
                st.write("<span style='color:red; font-size:24px;'>Atenção: Nota Inválida.</span>", unsafe_allow_html=True)
            else:
                st.write("<span style='color:green; font-size:24px;'>Seguro: Nota Válida.</span>", unsafe_allow_html=True)
        except requests.exceptions.RequestException as e:
            st.error(f"Erro na requisição: {e}")
