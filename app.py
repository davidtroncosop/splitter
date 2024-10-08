import streamlit as st
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import re
import gc

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
    Extrae los ítems, sus cantidades y sus precios unitarios en CLP (pesos chilenos).
    Devuelve SOLO un objeto JSON válido con la siguiente estructura:
    {
        "items": [
            {"item": "nombre del item", "cantidad": "cantidad", "precio_unitario": "precio unitario en formato CLP sin puntos"},
            ...
        ]
    }
    Asegúrate de devolver los precios como strings sin puntos de miles (ej. "5900" en lugar de "5.900").
    """
    respuesta = model.generate_content([prompt, imagen])
    del imagen  # Liberar la referencia a la imagen
    gc.collect()  # Forzar la recolección de basura
    return respuesta.text

def extraer_json(texto):
    json_match = re.search(r'\{.*\}', texto, re.DOTALL)
    return json_match.group() if json_match else None

@st.cache_data(max_entries=10, ttl=3600)
def procesar_resultado(resultado):
    json_str = extraer_json(resultado)
    if json_str:
        try:
            datos = json.loads(json_str)
            if 'items' in datos:
                for item in datos['items']:
                    item['cantidad'] = int(item['cantidad'])
                    item['precio_unitario'] = int(item['precio_unitario'])
                    item['total'] = item['cantidad'] * item['precio_unitario']
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
        match = re.search(r'([^:]+):\s*(\d+)\s*x\s*\$?\s*([\d.]+)', line)
        if match:
            item = match.group(1).strip()
            cantidad = int(match.group(2))
            precio_unitario = int(match.group(3).replace('.', ''))
            total = cantidad * precio_unitario
            items.append({"item": item, "cantidad": cantidad, "precio_unitario": precio_unitario, "total": total})
    
    return pd.DataFrame(items) if items else pd.DataFrame()

def procesar_items_por_lotes(df, nombres, num_personas, batch_size=10):
    matriz_consumo = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        for idx, row in batch.iterrows():
            st.write(f"**{row['item']}** (${row['total']:,.0f})")
            cols = st.columns(num_personas)
            fila_consumo = []
            for j, col in enumerate(cols):
                with col:
                    consumio = st.checkbox(f"{nombres[j]}", key=f"{nombres[j]}_{row['item']}_{idx}")
                    fila_consumo.append(1 if consumio else 0)
            matriz_consumo.append(fila_consumo)
    return np.array(matriz_consumo, dtype=np.float32)

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
            st.subheader("Tabla de ítems extraídos (puedes editar)")
            edited_df = st.data_editor(
                st.session_state.df,
                num_rows="dynamic",
                column_config={
                    "cantidad": st.column_config.NumberColumn(
                        "Cantidad",
                        help="Cantidad del ítem",
                        min_value=1,
                        step=1,
                    ),
                    "precio_unitario": st.column_config.NumberColumn(
                        "Precio Unitario",
                        help="Precio unitario en CLP",
                        min_value=0,
                        step=100,
                        format="$%d"
                    ),
                    "total": st.column_config.NumberColumn(
                        "Total",
                        help="Total por ítem",
                        format="$%d"
                    )
                },
                hide_index=True,
            )
            
            edited_df['total'] = edited_df['cantidad'] * edited_df['precio_unitario']
            total_items = edited_df['total'].sum()
            st.write(f"Total de los ítems: ${total_items:,.0f}")
            
            st.subheader("Asignación de consumo")
            num_personas = st.number_input("Número de personas", min_value=1, value=2, key="num_personas")
            
            nombres = []
            for i in range(num_personas):
                nombre = st.text_input(f"Nombre de la persona {i+1}", value=f"Persona {i+1}", key=f"nombre_{i}")
                nombres.append(nombre)
            
            st.write("Marca quién consumió cada ítem:")
            
            matriz_consumo = procesar_items_por_lotes(edited_df, nombres, num_personas)
            
            totales = np.array(edited_df['total'], dtype=np.float32)
            
            # Calcular el costo por ítem dividido entre las personas que lo consumieron
            costos_divididos = np.zeros_like(matriz_consumo, dtype=np.float32)
            for i, (total, consumidores) in enumerate(zip(totales, matriz_consumo)):
                num_consumidores = np.sum(consumidores)
                if num_consumidores > 0:
                    costos_divididos[i] = consumidores * (total / num_consumidores)
            
            gastos_por_persona = np.sum(costos_divididos, axis=0)
            
            st.subheader("Resultados")
            for nombre, gasto in zip(nombres, gastos_por_persona):
                st.write(f"{nombre} debe pagar: ${gasto:,.0f}")
            
            total = np.sum(gastos_por_persona)
            st.write(f"Total de la cuenta: ${total:,.0f}")

if __name__ == "__main__":
    main()