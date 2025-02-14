import streamlit as st
import pandas as pd
import requests

API_URL = "http://18.119.19.95/predict"  # Altere para o endpoint correto da sua API

def prediction():
    """
    Aba de Previsão de Fraude.

    Faz uma requisição à API para prever a probabilidade de fraude.
    """

    st.markdown("<h3 style='text-align: center;'>Previsão de Fraude</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>Utilize a API para prever a probabilidade de fraude em notas fiscais</h6>", unsafe_allow_html=True)

    # Formulário para inserção dos dados
    with st.form("fraud_prediction_form"):
        iss_retention = st.number_input("ISS Retention", value=0.0, format="%.2f")
        inss_tax_rate = st.number_input("INSS Tax Rate", value=0.0, format="%.2f")
        csll_tax_rate = st.number_input("CSLL Tax Rate", value=0.0, format="%.2f")
        calculated_value = st.number_input("Calculated Value", value=0.0, format="%.2f")
        cofins_tax_rate = st.number_input("COFINS Tax Rate", value=0.0, format="%.2f")
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
            if result["fraud"] == 1:
                st.write("<span style='color:red; font-size:24px;'>Atenção:</span> Nota Inválida.", unsafe_allow_html=True)
            else:
                st.write("<span style='color:green; font-size:24px;'>Seguro:</span> Nota Válida.", unsafe_allow_html=True)
        except requests.exceptions.RequestException as e:
            st.error(f"Erro na requisição: {e}")