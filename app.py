import streamlit as st
import os
import torch

# --- IMPORTACIONES REALES DE TUS COMPAÑEROS (SEMANA 3) ---
from src.physics.src2.audio_processing import extraer_audio_de_video, calcular_stft_y_mel, generar_mapa_calor_interactivo
from src.math_core.normalization import cargar_y_normalizar_audio, convertir_a_tensor_pytorch
from src.math_core.metrics import calcular_regularidad_fase
from src.ai.model import DeepfakeAudioCNN

# ==========================================
# 🧠 LÓGICA DE INFERENCIA IA 
# ==========================================
@st.cache_resource
def inicializar_red_neuronal():
    try:
        # 1. Instanciamos la arquitectura real que creó el Ing. en IA
        modelo = DeepfakeAudioCNN()
        
        # 2. Cargamos los pesos entrenados
        ruta_pesos = "src/ai/modelo_cnn.pth"
        if os.path.exists(ruta_pesos):
            modelo.load_state_dict(torch.load(ruta_pesos, map_location=torch.device('cpu')))
        else:
            st.warning("⚠️ Arquitectura de IA cargada, pero falta el archivo 'modelo_cnn.pth'. Se usará en modo no-entrenado temporalmente.")
            
        modelo.eval() # Apagamos el modo de entrenamiento para que solo haga inferencia
        return modelo
    except Exception as e:
        st.error(f"Error cargando el modelo de IA: {e}")
        return None

# 1. Configuración de la página
st.set_page_config(page_title="Detector Forense | UNESCO", page_icon="🔎", layout="wide")

st.title("🔎 Detector Forense de Audio 'Deepfake'")
st.markdown("### Plataforma de análisis espectral y métricas de fase - UNESCO Youth Hackathon 2026")
st.markdown("---") 

# Cargar la IA en la memoria RAM del servidor
modelo_ia = inicializar_red_neuronal()

# 2. Widget de Carga
st.markdown("### 📥 Ingesta de Datos")
archivo_video = st.file_uploader("Arrastra y suelta el video sospechoso aquí...", type=["mp4", "avi", "mov"])

if archivo_video is not None:
    st.success("✅ Video cargado exitosamente en la memoria del sistema.")
    st.video(archivo_video)
    
    st.markdown("---")
    st.markdown("### 🔬 Resultados del Análisis Forense")
    
    with st.spinner("⏳ Procesando señales y ejecutando inferencia profunda..."):
        
        # 1. Guardar archivo temporal del video
        ruta_temp_video = "temp_video.mp4"
        with open(ruta_temp_video, "wb") as f:
            f.write(archivo_video.getbuffer())
            
        try:
            # 2. PIPELINE DE SEÑALES REAL (El cableado final)
            ruta_audio = extraer_audio_de_video(ruta_temp_video, "temp_audio.wav")
            y, sr = cargar_y_normalizar_audio(ruta_audio)
            mel_db, mfccs = calcular_stft_y_mel(y, sr)
            
            # Fórmulas Matemáticas y Gráficos
            inestabilidad_fase = calcular_regularidad_fase(y)
            fig_mapa_calor = generar_mapa_calor_interactivo(mel_db, sr)
            
            # 3. LÓGICA DE INFERENCIA DE IA
            tensor_entrada = convertir_a_tensor_pytorch(ruta_audio)
            
            if modelo_ia is not None:
                with torch.no_grad():
                    # Añadimos la dimensión de lote (Batch) que espera PyTorch: [1, Canales, Alto, Ancho]
                    if tensor_entrada.dim() == 3:
                        tensor_entrada = tensor_entrada.unsqueeze(0)
                    
                    salida = modelo_ia(tensor_entrada)
                    
                    # Aplicamos Softmax para obtener el % exacto de probabilidad de la clase 1 (Deepfake)
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
    # RENDERIZADO EN PANTALLA
    # ==========================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🗺️ Mapa de Calor Espectral")
        # Aquí incrustamos mágicamente la figura interactiva de Plotly que hizo el Físico
        if fig_mapa_calor:
            st.plotly_chart(fig_mapa_calor, use_container_width=True)
        else:
            st.info("Esperando renderizado...")
        
    with col2:
        st.markdown("#### 📊 Métricas de Predicción")
        # Lógica de colores automáticos para advertir al usuario
        color_delta = "inverse" if porcentaje_ia > 50 else "normal"
        
        st.metric(label="🤖 Probabilidad de Deepfake (IA)", value=f"{porcentaje_ia:.2f}%", delta="Alta probabilidad de fraude" if porcentaje_ia > 50 else "Audio Auténtico", delta_color=color_delta)
        st.metric(label="📐 Inestabilidad de Fase", value=f"{inestabilidad_fase:.4f}", delta="Ruptura de fase detectada" if inestabilidad_fase > 0.03 else "Fase estable", delta_color=color_delta)
        st.metric(label="⚡ Energía en Altas Frecuencias", value="Procesado", delta="A la espera de estadística", delta_color="off")
        
    # Limpieza absoluta de la memoria local para proteger el servidor
    if os.path.exists(ruta_temp_video):
        os.remove(ruta_temp_video)
    if 'ruta_audio' in locals() and os.path.exists(ruta_audio):
        os.remove(ruta_audio)