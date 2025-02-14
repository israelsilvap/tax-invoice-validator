# Usar a imagem base do Python
FROM python:3.10

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar todos os arquivos do diretório atual para o contêiner
COPY . .

# Instalar as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta onde o uvicorn estará escutando
EXPOSE 8000

# Comando para rodar a API utilizando uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
