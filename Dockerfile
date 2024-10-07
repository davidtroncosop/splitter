# Usa la imagen oficial de Python como base
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos requeridos para la aplicación a /app
COPY . /app

# Instala las dependencias necesarias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir streamlit google-generativeai pillow python-dotenv pandas numpy

# Expone el puerto que utilizará Streamlit (el predeterminado es 8501)
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py"]
