import streamlit as st

# 1. Configuración de la página (Layout)
# layout="wide" hace que la aplicación use toda la pantalla en lugar de un recuadro pequeño
st.set_page_config(
    page_title="Detector Forense de Deepfakes | UNESCO",
    page_icon="🔎",
    layout="wide"
)

# 2. Títulos y Encabezados
st.title("🔎 Detector Forense de Audio 'Deepfake'")
st.markdown("### Plataforma de análisis espectral y métricas de fase - UNESCO Youth Hackathon 2026")
st.markdown("---")

st.write("Bienvenido. El sistema está en línea.")