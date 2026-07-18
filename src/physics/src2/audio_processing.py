import os
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')  # Evita que se abran ventanas flotantes de la GUI en servidores/Streamlit
import matplotlib.pyplot as plt

def extraer_audio_de_video(ruta_video, ruta_salida_audio="data/samples/temp_audio.wav"):
    """
    Separa el flujo de audio de un archivo de video y lo guarda en formato WAV.
    Compatible con MoviePy v2.0+
    """
    import moviepy as mp
    print(f"📦 Extrayendo audio de: {ruta_video}...")
    
    os.makedirs(os.path.dirname(ruta_salida_audio), exist_ok=True)
    
    video = mp.VideoFileClip(ruta_video)
    audio = video.audio
    audio.write_audiofile(ruta_salida_audio, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
    
    audio.close()
    video.close()   
    
    print(f"✅ Audio extraído exitosamente en: {ruta_salida_audio}")
    return ruta_salida_audio

def cargar_y_normalizar_audio(ruta_audio, target_sr=16000):
    """
    Carga un archivo de audio usando librosa, fuerza la monofonización
    y normaliza la amplitud pico a 1.0 para mantener la consistencia estadística.
    """
    print(f"💿 Cargando y preprocesando señal acústica a {target_sr} Hz...")
    y, sr = librosa.load(ruta_audio, sr=target_sr)
    
    if len(y) > 0:
        y = y / np.max(np.abs(y))
        
    return y, sr

def calcular_stft_y_mel(y, sr, n_fft=1024, hop_length=256):
    """
    Calcula la Transformada de Fourier de Tiempo Reducido (STFT)
    y la transforma a la Escala de Mel para análisis psicoacústico.
    """
    print("🔮 Calculando STFT y Espectrograma de Mel...")
    stft_compleja = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_magnitud = np.abs(stft_compleja)
    
    espectrograma_mel = librosa.feature.melspectrogram(
        S=stft_magnitud**2, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=128
    )
    
    mel_db = librosa.power_to_db(espectrograma_mel, ref=np.max)
    mfccs = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)
    
    return mel_db, mfccs

def aplicar_filtro_paso_alto(y, sr, cutoff_freq=4000.0, order=5):
    """
    Aplica un filtro digital Butterworth de tipo paso-alto para aislar
    las frecuencias superiores a 'cutoff_freq' (por defecto 4kHz).
    """
    print(f"🎛️ Aplicando filtro paso-alto Butterworth (Corte: {cutoff_freq} Hz, Orden: {order})...")
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y_filtrada = signal.filtfilt(b, a, y)
    
    return y_filtrada

def calcular_regularidad_fase(y, n_fft=1024, hop_length=256):
    """
    Calcula la métrica de regularidad de fase para analizar la continuidad de la onda.
    Devuelve la varianza de la derivada de la fase en el tiempo (métrica de inestabilidad).
    """
    print("📐 Analizando regularidad de la fase de la onda...")
    stft_compleja = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    fase = np.angle(stft_compleja)
    delta_fase = np.diff(fase, axis=1)
    delta_fase_envuelta = np.angle(np.exp(1j * delta_fase))
    varianza_fase = np.var(delta_fase_envuelta, axis=1)
    
    return float(np.mean(varianza_fase))

def convertir_a_tensor_pytorch(matriz, agregar_canal=True):
    """
    Convierte opcionalmente una matriz de características en un torch.Tensor listo para la CNN.
    """
    try:
        import torch
        print("🤖 Convirtiendo características a tensor de PyTorch...")
        tensor = torch.from_numpy(matriz).float()
        if agregar_canal:
            tensor = tensor.unsqueeze(0) 
        return tensor
    except ImportError:
        print("⚠️ PyTorch no detectado localmente. Retornando matriz NumPy original.")
        return matriz

def guardar_espectrograma_limpio(mel_db, ruta_salida):
    """
    Guarda el espectrograma de Mel como una imagen PNG completamente limpia (sin ejes ni bordes).
    Ideal para el procesamiento de la CNN.
    """
    print(f"🖼️ Guardando espectrograma limpio en: {ruta_salida}...")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    librosa.display.specshow(mel_db, cmap='magma')
    plt.savefig(ruta_salida, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print("✅ Imagen limpia guardada exitosamente.")

def guardar_espectrograma_forense(mel_db, sr, hop_length=256, ruta_salida="data/samples/espectrograma_forense.png"):
    """
    Genera y guarda un espectrograma de diagnóstico con ejes, etiquetas y escala de decibelios.
    """
    print(f"📊 Generando espectrograma forense en: {ruta_salida}...")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    plt.figure(figsize=(10, 4.5))
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title("Análisis Espectral de Voz (Mapa Forense de Calor)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Tiempo (segundos)", fontsize=10)
    plt.ylabel("Frecuencia Psicoacústica (Escala Mel)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print("✅ Espectrograma forense guardado.")

def contrastar_espectrogramas_forenses(ruta_real, ruta_falso, target_sr=16000):
    """
    Compara propiedades físicas espectrales, de energía y de fase entre muestras reales y sospechosas.
    """
    print("\n🔬 Iniciando análisis comparativo forense...")
    y_real, sr_real = cargar_y_normalizar_audio(ruta_real, target_sr=target_sr)
    y_falso, sr_falso = cargar_y_normalizar_audio(ruta_falso, target_sr=target_sr)
    
    mel_db_real, mfccs_real = calcular_stft_y_mel(y_real, sr_real)
    mel_db_falso, mfccs_falso = calcular_stft_y_mel(y_falso, sr_falso)
    
    fase_real = calcular_regularidad_fase(y_real)
    fase_falso = calcular_regularidad_fase(y_falso)
    
    y_altas_real = aplicar_filtro_paso_alto(y_real, sr_real, cutoff_freq=4000.0)
    y_altas_falso = aplicar_filtro_paso_alto(y_falso, sr_falso, cutoff_freq=4000.0)
    
    energia_total_real = np.sum(y_real**2)
    energia_alta_real = np.sum(y_altas_real**2)
    ratio_alta_real = (energia_alta_real / energia_total_real) * 100 if energia_total_real > 0 else 0
    
    energia_total_falso = np.sum(y_falso**2)
    energia_alta_falso = np.sum(y_altas_falso**2)
    ratio_alta_falso = (energia_alta_falso / energia_total_falso) * 100 if energia_total_falso > 0 else 0
    
    min_frames = min(mfccs_real.shape[1], mfccs_falso.shape[1])
    distancia_mfcc = np.linalg.norm(mfccs_real[:, :min_frames] - mfccs_falso[:, :min_frames], axis=0)
    disparidad_timbrica_media = float(np.mean(distancia_mfcc))
    
    reporte_contraste = {
        "fase": {
            "real": fase_real,
            "falso": fase_falso,
            "diferencia_absoluta": abs(fase_real - fase_falso)
        },
        "energia_alta_frecuencia": {
            "real_porcentaje": ratio_alta_real,
            "falso_porcentaje": ratio_alta_falso,
            "desviacion_vocoder": abs(ratio_alta_real - ratio_alta_falso)
        },
        "disparidad_timbrica_mfcc": disparidad_timbrica_media,
        "matrices": {
            "mel_db_real": mel_db_real,
            "mel_db_falso": mel_db_falso
        }
    }
    print("✅ Análisis comparativo completado con éxito.")
    return reporte_contraste

def generar_mapa_calor_interactivo(mel_db, sr, hop_length=256):
    """
    Genera un Heatmap interactivo utilizando Plotly para ser embebido en Streamlit.
    """
    try:
        import plotly.graph_objects as go
        print("📊 Diseñando mapa de calor interactivo con Plotly...")
        times = librosa.times_like(mel_db, sr=sr, hop_length=hop_length)
        mel_frequencies = librosa.mel_frequencies(n_mels=mel_db.shape[0], fmax=sr//2)
        
        fig = go.Figure(data=go.Heatmap(
            z=mel_db, x=times, y=mel_frequencies,
            colorscale='Magma', zmin=-80, zmax=0,
            colorbar=dict(title="Potencia (dB)")
        ))
        
        fig.update_layout(
            title=dict(text="<b>Firma Espectral Interactiva (Auditoría Forense)</b>", font=dict(size=16)),
            xaxis=dict(title="Tiempo (segundos)"),
            yaxis=dict(title="Frecuencia Psicoacústica (Hz - Escala Mel)"),
            margin=dict(l=50, r=50, b=50, t=50), height=400, template="plotly_dark"
        )
        return fig
    except ImportError:
        print("⚠️ Plotly no instalado. Se omite el mapa interactivo.")
        return None

def generar_espectrograma_forense_web(mel_db, sr, hop_length=256):
    """
    Genera un espectrograma de diagnóstico y devuelve el objeto Figure de Matplotlib para la UI.
    """
    print("📊 Generando objeto de espectrograma forense optimizado para la web...")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Análisis Espectral de Voz (Mapa Forense de Calor)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Tiempo (segundos)", fontsize=10)
    ax.set_ylabel("Frecuencia Psicoacústica (Escala Mel)", fontsize=10)
    fig.tight_layout()
    return fig

# --- BLOQUE DE PRUEBA LOCAL CON RUTAS RELATIVAS ---
if __name__ == "__main__":
    print("\n--- 🔬 PIPELINE LOCAL DE INGENIERÍA DE SEÑALES ---")
    
    # Conseguir la raíz del proyecto dinámicamente de forma relativa
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
    
    # Configurar las rutas relativas en el entorno
    ruta_video_test = os.path.join(DATA_DIR, "test_video.mp4")
    ruta_wav_real = os.path.join(DATA_DIR, "temp_audio_real.wav")
    ruta_wav_falso = os.path.join(DATA_DIR, "temp_audio_falso.wav")
    ruta_img_clean = os.path.join(DATA_DIR, "test_espectrograma_clean.png")
    ruta_img_forense = os.path.join(DATA_DIR, "test_espectrograma_forense.png")

    # Crear directorios simulados si no existen en el clon local para evitar crash
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ruta_video_test):
        print(f"⚠️ Coloca un video de prueba en: {ruta_video_test} para correr el test completo.")
    else:
        try:
            extraer_audio_de_video(ruta_video_test, ruta_wav_real)
            if not os.path.exists(ruta_wav_falso):
                import shutil
                shutil.copy(ruta_wav_real, ruta_wav_falso)

            y, sr = cargar_y_normalizar_audio(ruta_wav_real, target_sr=16000)
            mel_db, mfccs = calcular_stft_y_mel(y, sr)
            
            mel_tensor = convertir_a_tensor_pytorch(mel_db)
            guardar_espectrograma_limpio(mel_db, ruta_img_clean)
            guardar_espectrograma_forense(mel_db, sr, ruta_salida=ruta_img_forense)
            
            resultado = contrastar_espectrogramas_forenses(ruta_wav_real, ruta_wav_falso)
            print("\n✅ Test de Pipeline Modularizado Ejecutado Correctamente.")
        except Exception as e:
            print(f"❌ Error en el test local: {e}")
