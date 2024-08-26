# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos de requerimientos, setup y modelos
COPY requirements.txt .
COPY setup.py .
COPY models/ ./models/

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Instalar la librería local usando setup.py
RUN pip install -e .

# Instalar el modelo de español para spaCy
RUN python -m spacy download es_core_news_sm

# Copiar el código de la aplicación y los módulos necesarios
COPY app/ ./app/
COPY src/ ./src/

# Exponer el puerto en el que se ejecutará Flask
EXPOSE 7000

# Comando para ejecutar la aplicación
CMD ["python", "app/app.py"]