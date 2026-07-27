import sys
import os
import uuid

# 1. FUERZA A PYTHON A RECONOCER LA CARPETA RAÍZ Y EL PAQUETE /src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch

# --- IMPORTACIONES MODULARES DE TU EQUIPO ---
from src.physics.audio_processing import (
    extraer_audio_de_video, calcular_stft_y_mel, generar_mapa_calor_interactivo,
    cargar_y_normalizar_audio, convertir_a_tensor_pytorch,
    predecir_audio_completo
)
from src.math_core.metrics import calcular_regularidad_fase
from src.ai.model import DeepfakeAudioCNN
from src.ai.custom_dataset import DeepfakeAudioDataset
from src.utils.social_downloader import descargar_video_de_red_social
import numpy as np

# Constante de frecuencia de muestreo unificada (Estándar 16kHz optimizado)
TARGET_SR = 16000

# ==========================================
# 🧠 LÓGICA DE INICIALIZACIÓN DE LA IA
# ==========================================
def inicializar_red_neuronal():
    try:
        modelo = DeepfakeAudioCNN()
        ruta_pesos = "src/ai/best_model.pth" 
        
        if os.path.exists(ruta_pesos):
            modelo.load_state_dict(torch.load(ruta_pesos, map_location=torch.device('cpu')))
            print("✅ PESOS CARGADOS CORRECTAMENTE")
        else:
            print(f"🚨 ERROR CRÍTICO: No se encontró {ruta_pesos}")
            st.error(f"🚨 ERROR: El archivo de IA ({ruta_pesos}) no existe. El modelo está adivinando al azar.")
            
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

# ==========================================
# 🧠 FUNCIÓN COMPARTIDA DE PROCESAMIENTO + RENDERIZADO
# ==========================================
def _procesar_y_mostrar(ruta_video_o_audio, ruta_audio, nombre_original, es_audio):
    """Procesa un archivo local (subido o descargado) y renderiza resultados."""
    archivos_limpiar = [ruta_audio]
    if not es_audio and ruta_video_o_audio != ruta_audio:
        archivos_limpiar.append(ruta_video_o_audio)

    st.markdown("---")
    st.markdown("### 🔬 Resultados del Análisis Forense")

    with st.spinner("⏳ Procesando señales y ejecutando inferencia profunda..."):
        os.makedirs("data", exist_ok=True)

        inestabilidad_fase = 0.0
        porcentaje_ia = 0.0
        fig_mapa_calor = None
        predicciones_por_ventana = []
        tiempos_ventanas = []

        try:
            if not es_audio:
                extraer_audio_de_video(ruta_video_o_audio, ruta_audio)

            if not os.path.exists(ruta_audio):
                raise Exception("El sistema no pudo extraer la pista de audio. Verifica que el video no sea mudo.")

            y, sr = cargar_y_normalizar_audio(ruta_audio, target_sr=TARGET_SR)
            mel_db, mfccs, stft_compleja = calcular_stft_y_mel(y, sr)

            inestabilidad_fase = calcular_regularidad_fase(y)
            fig_mapa_calor, warning_msg = generar_mapa_calor_interactivo(mel_db, sr, y_audio=y, stft_compleja=stft_compleja)

            if warning_msg:
                st.warning(warning_msg)

            if modelo_ia is not None:
                porcentaje_ia, predicciones_por_ventana, tiempos_ventanas = predecir_audio_completo(
                    mel_db, modelo_ia, device=torch.device("cpu")
                )

        except Exception as e:
            st.error(f"Error crítico en el procesamiento: {e}")

        finally:
            for p in archivos_limpiar:
                if os.path.exists(p):
                    try:
                        os.remove(p)
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

        if porcentaje_ia > 85:
            estado_alerta = "🚨 ALTO RIESGO: Fraude Sintético"
            color_delta = "inverse"
        elif porcentaje_ia > 60:
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
            if len(predicciones_por_ventana) > 1:
                st.write(f"**Ventanas analizadas:** {len(predicciones_por_ventana)} (audio completo, sin truncar)")
                for t_ini, prob in zip(tiempos_ventanas, predicciones_por_ventana):
                    st.write(f"  • Segundo {t_ini:.1f}s: {prob:.1f}% Fake")

        st.divider()

        st.metric(
            label="📐 Inestabilidad de Fase",
            value=f"{inestabilidad_fase:.4f}",
            delta="Ruptura de fase detectada" if inestabilidad_fase > 3.5 else "Fase estable",
            delta_color="inverse" if inestabilidad_fase > 3.5 else "normal"
        )


# ==========================================
# 📁 PESTAÑAS DE INGESTA
# ==========================================
tab_upload, tab_link = st.tabs(["📁 Subir archivo", "🔗 Pegar enlace (Facebook / Instagram / TikTok / X)"])

with tab_upload:
    archivo_video = st.file_uploader("Arrastra y suelta el video o audio sospechoso aquí...", type=["mp4", "avi", "mov", "wav", "mp3"])

    if archivo_video is not None:
        if st.session_state.ultimo_archivo != archivo_video.name:
            st.session_state.ultimo_archivo = archivo_video.name

        st.success(f"✅ Archivo '{archivo_video.name}' cargado exitosamente.")
        st.video(archivo_video)

        id_unico = str(uuid.uuid4())[:8]
        ruta_temp_video = f"data/temp_video_{id_unico}.mp4"
        ruta_audio = f"data/temp_audio_{id_unico}.wav"

        nombre_archivo = archivo_video.name.lower()
        es_audio = nombre_archivo.endswith(".wav") or nombre_archivo.endswith(".mp3")

        if es_audio:
            with open(ruta_audio, "wb") as f:
                f.write(archivo_video.getbuffer())
            _procesar_y_mostrar(ruta_audio, ruta_audio, archivo_video.name, True)
        else:
            with open(ruta_temp_video, "wb") as f:
                f.write(archivo_video.getbuffer())
            _procesar_y_mostrar(ruta_temp_video, ruta_audio, archivo_video.name, False)

with tab_link:
    url = st.text_input(
        "Pega el enlace del video:",
        placeholder="https://www.tiktok.com/@usuario/video/123456789..."
    )

    if url and st.button("Analizar enlace", type="primary"):
        try:
            with st.spinner("⏳ Descargando contenido desde la red social..."):
                ruta_descargada = descargar_video_de_red_social(url)

            nombre_original = os.path.basename(ruta_descargada)
            st.success(f"✅ Video descargado exitosamente.")
            st.video(ruta_descargada)

            id_unico = str(uuid.uuid4())[:8]
            ruta_audio = f"data/temp_audio_{id_unico}.wav"
            _procesar_y_mostrar(ruta_descargada, ruta_audio, nombre_original, False)

        except ValueError as e:
            st.error(str(e))
        except RuntimeError as e:
            st.error(str(e))
