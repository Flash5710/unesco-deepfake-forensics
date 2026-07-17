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
    
    # [ESPACIO RESERVADO PARA LA INTEGRACIÓN DE LA SEMANA 3]
    st.info("⏳ Conectando con el pipeline de ingeniería de señales...")