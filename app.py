import streamlit as st

# 1. Configuración de la página (Layout)
st.set_page_config(
    page_title="Detector Forense de Deepfakes | UNESCO",
    page_icon="🔎",
    layout="wide"
)

# 2. Títulos y Encabezados
st.title("🔎 Detector Forense de Audio 'Deepfake'")
st.markdown("### Plataforma de análisis espectral y métricas de fase - UNESCO Youth Hackathon 2026")
st.markdown("---") 

# 3. Widget de Carga de Archivos (Drag-and-Drop) - Tarea del Martes
st.markdown("### 📥 Ingesta de Datos")
archivo_video = st.file_uploader(
    "Arrastra y suelta el video sospechoso aquí para iniciar el análisis forense", 
    type=["mp4", "avi", "mov"]
)

# 4. Lógica de Interfaz (¿Qué pasa cuando el usuario suelta un video?)
if archivo_video is not None:
    st.success("✅ Video cargado exitosamente en la memoria del sistema.")
    
    # Reproductor visual para que el usuario confirme qué video subió
    st.video(archivo_video)
    
    st.markdown("---")
    st.markdown("### 🔬 Resultados del Análisis Forense")
    
    # --- MAQUETACIÓN ---
    # Dividimos la pantalla en 2 columnas (la proporción [2, 1] hace que la izquierda sea el doble de ancha)
    col1, col2 = st.columns([2, 1])
    
    # Contenedor del Físico (Mapa de calor)
    with col1:
        st.markdown("#### 🗺️ Mapa de Calor Espectral")
        # Aquí incrustaremos: st.plotly_chart(generar_mapa_calor_interactivo(...))
        st.info("El mapa interactivo de frecuencias se renderizará en este espacio.")
        
    # Contenedores del Matemático y la IA (Métricas numéricas)
    with col2:
        st.markdown("#### 📊 Métricas de Predicción")
        # st.metric es un widget nativo para mostrar números grandes e impactantes
        
        # 1. Porcentaje de la Red Neuronal (IA)
        st.metric(
            label="🤖 Probabilidad de Deepfake (IA)", 
            value="-- %", 
            delta="Esperando predicción..."
        )
        
        # 2. Tu métrica matemática de fase
        st.metric(
            label="📐 Inestabilidad de Fase", 
            value="--", 
            delta="Esperando análisis...",
            delta_color="off"
        )
        
        # 3. La anomalía de energía del Físico
        st.metric(
            label="⚡ Energía en Altas Frecuencias", 
            value="-- %", 
            delta="Esperando filtro...",
            delta_color="off"
        )