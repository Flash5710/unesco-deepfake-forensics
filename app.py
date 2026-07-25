import streamlit as st
import os
import torch

# --- IMPORTACIONES REALES DE TUS COMPAÑEROS ---
from src.physics.src2.audio_processing import extraer_audio_de_video, calcular_stft_y_mel, generar_mapa_calor_interactivo, TARGET_SR
from src.math_core.normalization import cargar_y_normalizar_audio, convertir_a_tensor_pytorch
from src.math_core.metrics import calcular_regularidad_fase
from src.ai.model import DeepfakeAudioCNN

# ==========================================
# 🧠 LÓGICA DE INFERENCIA IA 
# ==========================================
@st.cache_resource
def inicializar_red_neuronal():
    try:
        modelo = DeepfakeAudioCNN()
        ruta_pesos = "src/ai/modelo_cnn.pth"
        if os.path.exists(ruta_pesos):
            modelo.load_state_dict(torch.load(ruta_pesos, map_location=torch.device('cpu')))
        modelo.eval()
        return modelo
    except Exception as e:
        st.error(f"Error cargando el modelo de IA: {e}")
        return None

st.set_page_config(page_title="Detector Forense | UNESCO", page_icon="🔎", layout="wide")

st.title("🔎 Detector Forense de Audio 'Deepfake'")
st.markdown("### Plataforma de análisis espectral y métricas de fase - UNESCO Youth Hackathon 2026")
st.markdown("---") 

modelo_ia = inicializar_red_neuronal()

st.markdown("### 📥 Ingesta de Datos")
archivo_video = st.file_uploader("Arrastra y suelta el video sospechoso aquí...", type=["mp4", "avi", "mov"])

if archivo_video is not None:
    st.success("✅ Video cargado exitosamente.")
    st.video(archivo_video)
    
    st.markdown("---")
    st.markdown("### 🔬 Resultados del Análisis Forense")
    
    with st.spinner("⏳ Procesando señales y ejecutando inferencia profunda..."):
        ruta_temp_video = "temp_video.mp4"
        with open(ruta_temp_video, "wb") as f:
            f.write(archivo_video.getbuffer())
            
        try:
            ruta_audio = extraer_audio_de_video(ruta_temp_video, "temp_audio.wav")
            
            # ⚠️ Parche aplicado: Usamos TARGET_SR en vez de hardcodear números
            y, sr = cargar_y_normalizar_audio(ruta_audio, target_sr=TARGET_SR)
            
            # ⚠️ Parche aplicado: Desempaquetamos los 3 valores (mel_db, mfccs, stft_compleja)
            mel_db, mfccs, stft_compleja = calcular_stft_y_mel(y, sr)
            
            inestabilidad_fase = calcular_regularidad_fase(y)
            fig_mapa_calor = generar_mapa_calor_interactivo(mel_db, sr)
            
            tensor_entrada = convertir_a_tensor_pytorch(ruta_audio)
            
            if modelo_ia is not None:
                with torch.no_grad():
                    if tensor_entrada.dim() == 3:
                        tensor_entrada = tensor_entrada.unsqueeze(0)
                    salida = modelo_ia(tensor_entrada)
                    probabilidades = torch.nn.functional.softmax(salida, dim=1)
                    porcentaje_ia = probabilidades[0][1].item() * 100 
            else:
                porcentaje_ia = 0.0 
        
        except Exception as e:
            st.error(f"Error crítico en el procesamiento: {e}")
            inestabilidad_fase = 0.0
            porcentaje_ia = 0.0
            fig_mapa_calor = None

    # ==========================================
    # INTEGRACIÓN COMPLETA DE UI
    # ==========================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🗺️ Mapa de Calor Espectral")
        if fig_mapa_calor:
            st.plotly_chart(fig_mapa_calor, use_container_width=True)
        else:
            st.info("Esperando renderizado...")
        
    with col2:
        st.markdown("#### 📊 Métricas de Predicción")
        
        # 1. Lógica dinámica de niveles de alerta para la IA
        if porcentaje_ia > 75:
            estado_alerta = "🚨 ALTO RIESGO: Fraude Sintético"
            color_delta = "inverse"
        elif porcentaje_ia > 40:
            estado_alerta = "⚠️ ADVERTENCIA: Audio Modificado"
            color_delta = "off"
        else:
            estado_alerta = "✅ SEGURO: Audio Auténtico"
            color_delta = "normal"
            
        st.metric(label="🤖 Probabilidad de Deepfake (IA)", 
                  value=f"{porcentaje_ia:.1f}%", 
                  delta=estado_alerta, 
                  delta_color=color_delta)
                  
        # 2. Barra de progreso visual para fácil lectura
        st.progress(int(porcentaje_ia) / 100)
        
        # 3. Contenedor expansible con metadatos técnicos (Requisito de UI)
        with st.expander("Ver desglose técnico de inferencia"):
            st.write(f"**Arquitectura:** DeepfakeAudioCNN")
            st.write(f"**Tasa de Muestreo:** {TARGET_SR} Hz")
            st.write(f"**Tensores Procesados:** MFCCs + STFT Compleja")

        st.divider()

        # Métricas matemáticas
        st.metric(label="📐 Inestabilidad de Fase", 
                  value=f"{inestabilidad_fase:.4f}", 
                  delta="Ruptura de fase detectada" if inestabilidad_fase > 0.03 else "Fase estable", 
                  delta_color="inverse" if inestabilidad_fase > 0.03 else "normal")
        
    # Limpieza
    if os.path.exists(ruta_temp_video):
        os.remove(ruta_temp_video)
    if 'ruta_audio' in locals() and os.path.exists(ruta_audio):
        os.remove(ruta_audio)