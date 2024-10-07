import streamlit as st
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import re

# Cargar variables de entorno
load_dotenv()

# Configurar la API key
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

# Cargar el modelo
model = genai.GenerativeModel('gemini-1.5-flash')

def cargar_imagen(uploaded_file):
    return PIL.Image.open(uploaded_file)

def analizar_imagen(imagen):
    prompt = """
    Analiza esta imagen de una boleta de restaurante. 
    Extrae los ítems y sus precios en CLP (pesos chilenos).
    Devuelve SOLO un objeto JSON válido con la siguiente estructura:
    {
        "items": [
            {"item": "nombre del item", "precio": "precio en formato CLP sin puntos"},
            ...
        ]
    }
    Asegúrate de devolver los precios como strings sin puntos de miles (ej. "5900" en lugar de "5.900").
    """
    respuesta = model.generate_content([prompt, imagen])
    return respuesta.text

def extraer_json(texto):
    json_match = re.search(r'\{.*\}', texto, re.DOTALL)
    return json_match.group() if json_match else None

def procesar_resultado(resultado):
    st.text("Respuesta completa del modelo:")
    st.text(resultado)
    
    json_str = extraer_json(resultado)
    if json_str:
        try:
            datos = json.loads(json_str)
            if 'items' in datos:
                for item in datos['items']:
                    item['precio'] = int(item['precio'])
                return pd.DataFrame(datos['items'])
            else:
                st.error("El JSON no contiene la clave 'items' esperada.")
        except json.JSONDecodeError as e:
            st.error(f"Error al decodificar JSON: {e}")
    else:
        st.error("No se pudo extraer un objeto JSON válido de la respuesta.")
    
    # Procesamiento manual en caso de error
    st.warning("Intentando procesar manualmente la respuesta...")
    items = []
    for line in resultado.split('\n'):
        match = re.search(r'([^:]+):\s*\$?\s*([\d.]+)', line)
        if match:
            precio = int(match.group(2).replace('.', ''))
            items.append({"item": match.group(1).strip(), "precio": precio})
    
    return pd.DataFrame(items) if items else pd.DataFrame()

def main():
    st.title("Divisor de Gastos de Restaurante")
    
    uploaded_file = st.file_uploader("Carga la imagen de la boleta", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        if 'imagen_procesada' not in st.session_state:
            st.session_state.imagen_procesada = False

        imagen = cargar_imagen(uploaded_file)
        st.image(imagen, caption='Boleta cargada', use_column_width=True)
        
        if not st.session_state.imagen_procesada:
            with st.spinner('Procesando la imagen...'):
                try:
                    resultado = analizar_imagen(imagen)
                    df = procesar_resultado(resultado)
                    if not df.empty:
                        st.session_state.df = df
                        st.session_state.imagen_procesada = True
                    else:
                        st.error("No se pudo procesar la imagen correctamente.")
                        return
                except Exception as e:
                    st.error(f"Error inesperado al procesar la imagen: {str(e)}")
                    return

        if st.session_state.imagen_procesada:
            st.subheader("Tabla de ítems extraídos (puedes editar los precios)")
            edited_df = st.data_editor(
                st.session_state.df,
                num_rows="dynamic",
                column_config={
                    "precio": st.column_config.NumberColumn(
                        "Precio",
                        help="Precio en CLP",
                        min_value=0,
                        max_value=1000000,
                        step=100,
                        format="$%d"
                    )
                }
            )
            
            total_items = edited_df['precio'].sum()
            st.write(f"Total de los ítems: ${total_items:,.0f}")
            
            st.subheader("Matriz de Consumo")
            num_personas = st.number_input("Número de personas", min_value=1, value=2, key="num_personas")
            
            nombres = [st.text_input(f"Nombre de la persona {i+1}", value=f"Persona {i+1}", key=f"nombre_{i}") for i in range(num_personas)]
            
            matriz_consumo = []
            for idx, item in enumerate(edited_df['item']):
                row = []
                cols = st.columns(num_personas)
                for i, col in enumerate(cols):
                    with col:
                        key_checkbox = f"{nombres[i]}_{item}_{idx}"
                        consumio = st.checkbox(f"{nombres[i]} consumió {item}", key=key_checkbox)
                        row.append(1 if consumio else 0)
                matriz_consumo.append(row)
            
            matriz_consumo = np.array(matriz_consumo)
            precios = np.array(edited_df['precio'])
            
            # Calcular el costo por ítem dividido entre las personas que lo consumieron
            costos_divididos = np.zeros_like(matriz_consumo, dtype=float)
            for i, (precio, consumidores) in enumerate(zip(precios, matriz_consumo)):
                num_consumidores = np.sum(consumidores)
                if num_consumidores > 0:
                    costos_divididos[i] = consumidores * (precio / num_consumidores)
            
            gastos_por_persona = np.sum(costos_divididos, axis=0)
            
            st.subheader("Resultados")
            for nombre, gasto in zip(nombres, gastos_por_persona):
                st.write(f"{nombre} debe pagar: ${gasto:,.0f}")
            
            total = np.sum(gastos_por_persona)
            st.write(f"Total de la cuenta: ${total:,.0f}")

if __name__ == "__main__":
    main()