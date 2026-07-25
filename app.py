import sys
import os
import uuid

# 1. FUERZA A PYTHON A RECONOCER LA CARPETA RAÍZ Y EL PAQUETE /src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch

# --- IMPORTACIONES MODULARES DE TU EQUIPO ---
from src.physics.audio_processing import extraer_audio_de_video, calcular_stft_y_mel, generar_mapa_calor_interactivo
from src.math_core.normalization import cargar_y_normalizar_audio, convertir_a_tensor_pytorch
from src.math_core.metrics import calcular_regularidad_fase
from src.ai.model import DeepfakeAudioCNN

# Constante de frecuencia de muestreo unificada
TARGET_SR = 16000

# ==========================================
# 🧠 LÓGICA DE INICIALIZACIÓN DE LA IA
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

# ==========================================
# 🎨 CONFIGURACIÓN DE LA INTERFAZ (UI)
# ==========================================
st.set_page_config(page_title="Detector Forense | UNESCO", page_icon="🔎", layout="wide")

st.title("🔎 Detector Forense de Audio 'Deepfake'")
st.markdown("### Plataforma de análisis espectral y métricas de fase - UNESCO Youth Hackathon 2026")
st.markdown("---") 

modelo_ia = inicializar_red_neuronal()

# Control de estado de sesión para reactividad continua
if 'ultimo_archivo' not in st.session_state:
    st.session_state.ultimo_archivo = None

st.markdown("### 📥 Ingesta de Datos")
archivo_video = st.file_uploader("Arrastra y suelta el video sospechoso aquí...", type=["mp4", "avi", "mov"])

if archivo_video is not None:
    if st.session_state.ultimo_archivo != archivo_video.name:
        st.session_state.ultimo_archivo = archivo_video.name
        
    st.success(f"✅ Video '{archivo_video.name}' cargado exitosamente.")
    st.video(archivo_video)
    
    st.markdown("---")
    st.markdown("### 🔬 Resultados del Análisis Forense")
    
    with st.spinner("⏳ Procesando señales y ejecutando inferencia profunda..."):
        # Aseguramos que la carpeta data exista
        os.makedirs("data", exist_ok=True)
        
        # Identificador único para evitar bloqueos I/O en Windows
        id_unico = str(uuid.uuid4())[:8]
        ruta_temp_video = f"data/temp_video_{id_unico}.mp4"
        ruta_audio = f"data/temp_audio_{id_unico}.wav"
        
        try:
            # Guardar el buffer del video
            with open(ruta_temp_video, "wb") as f:
                f.write(archivo_video.getbuffer())
                
            # 1. Extracción de señal de audio
            extraer_audio_de_video(ruta_temp_video, ruta_audio)
            
            # Escudo de validación
            if not os.path.exists(ruta_audio):
                raise Exception("El sistema no pudo extraer la pista de audio. Verifica que el video no sea mudo.")
                
            # 2. Carga y Normalización (Núcleo Matemático + Físico)
            y, sr = cargar_y_normalizar_audio(ruta_audio, target_sr=TARGET_SR)
            mel_db, mfccs, stft_compleja = calcular_stft_y_mel(y, sr)
            
            # 3. Métricas y Gráficas
            inestabilidad_fase = calcular_regularidad_fase(y)
            fig_mapa_calor, warning_msg = generar_mapa_calor_interactivo(mel_db, sr, y_audio=y, stft_compleja=stft_compleja)
            
            if warning_msg:
                st.warning(warning_msg)
            
            # 4. Inferencia con la Red Neuronal (CNN)
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
            
        finally:
            # Limpieza garantizada de basura en disco
            if os.path.exists(ruta_temp_video):
                try:
                    os.remove(ruta_temp_video)
                except:
                    pass
            if os.path.exists(ruta_audio):
                try:
                    os.remove(ruta_audio)
                except:
                    pass

    # ==========================================
    # 📊 CONTENEDORES Y RENDERIZADO VISUAL
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
        
        if porcentaje_ia > 75:
            estado_alerta = "🚨 ALTO RIESGO: Fraude Sintético"
            color_delta = "inverse"
        elif porcentaje_ia > 40:
            estado_alerta = "⚠️ ADVERTENCIA: Audio Modificado"
            color_delta = "off"
        else:
            estado_alerta = "✅ SEGURO: Audio Auténtico"
            color_delta = "normal"
            
        st.metric(
            label="🤖 Probabilidad de Deepfake (IA)", 
            value=f"{porcentaje_ia:.1f}%", 
            delta=estado_alerta, 
            delta_color=color_delta
        )
        st.progress(int(porcentaje_ia) / 100)
        
        with st.expander("Ver desglose técnico de inferencia"):
            st.write("**Arquitectura:** DeepfakeAudioCNN")
            st.write(f"**Tasa de Muestreo:** {TARGET_SR} Hz")
            st.write("**Tensores Procesados:** MFCCs + STFT Compleja")

        st.divider()

        st.metric(
            label="📐 Inestabilidad de Fase", 
            value=f"{inestabilidad_fase:.4f}", 
            delta="Ruptura de fase detectada" if inestabilidad_fase > 0.03 else "Fase estable", 
            delta_color="inverse" if inestabilidad_fase > 0.03 else "normal"
        )