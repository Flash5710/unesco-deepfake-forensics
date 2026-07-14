import os
import numpy as np
import librosa
import moviepy as mp
import scipy.signal as signal
import matplotlib.pyplot as plt

def extraer_audio_de_video(ruta_video, ruta_salida_audio="data/samples/temp_audio.wav"):
    """
    Separa el flujo de audio de un archivo de video y lo guarda en formato WAV.
    Compatible con MoviePy v2.0+
    """
    print(f"📦 Extrayendo audio de: {ruta_video}...")
    
    # Asegurar que el directorio de salida exista
    os.makedirs(os.path.dirname(ruta_salida_audio), exist_ok=True)
    
    # Cargar el video y extraer el canal de audio
    video = mp.VideoFileClip(ruta_video)
    audio = video.audio
    
    # Escribir el archivo de audio en el disco
    # Usamos fps=16000 o 22050 para forzar el remuestreo desde la extracción
    audio.write_audiofile(ruta_salida_audio, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
    
    # Cerrar los descriptores de archivo para liberar memoria ram
    audio.close()
    video.close()
    
    print(f"✅ Audio extraído exitosamente en: {ruta_salida_audio}")
    return ruta_salida_audio

def calcular_stft_y_mel(y, sr, n_fft=1024, hop_length=256):
    """
    Calcula la Transformada de Fourier de Tiempo Reducido (STFT)
    y la transforma a la Escala de Mel para análisis psicoacústico.
    """
    print("🔮 Calculando STFT y Espectrograma de Mel...")
    
    # 1. Calcular la STFT (Magnitud del espectro a lo largo del tiempo)
    # Usamos una ventana de Hanning implícita en librosa
    stft_compleja = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_magnitud = np.abs(stft_compleja)
    
    # 2. Convertir a Espectrograma de Mel
    # n_mibs=128 distribuye las frecuencias emulando la percepción humana
    espectrograma_mel = librosa.feature.melspectrogram(
        S=stft_magnitud**2, # Usamos potencia espectral
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=128
    )
    
    # 3. Convertir a escala logarítmica (Decibelios)
    # Los artefactos de los vocoders suelen ser sutiles; la escala logarítmica los resalta
    mel_db = librosa.power_to_db(espectrograma_mel, ref=np.max)
    
    # 4. Extraer coeficientes MFCC (Métricas estadísticas clave para nuestro Matemático)
    mfccs = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)
    
    return mel_db, mfccs
def aplicar_filtro_paso_alto(y, sr, cutoff_freq=4000.0, order=5):
    """
    Aplica un filtro digital Butterworth de tipo paso-alto para aislar
    las frecuencias superiores a 'cutoff_freq' (por defecto 4kHz).
    Ayuda a exponer artefactos y ruido espectral de los vocoders de IA.
    """
    print(f"🎛️  Aplicando filtro paso-alto Butterworth (Corte: {cutoff_freq} Hz, Orden: {order})...")
    
    # 1. Calcular la frecuencia de Nyquist (Límite físico del sistema analizado)
    nyquist = 0.5 * sr
    
    # 2. Normalizar la frecuencia de corte según Nyquist (Requisito de Scipy)
    normal_cutoff = cutoff_freq / nyquist
    
    # 3. Diseñar el filtro (b y a son los coeficientes de la función de transferencia)
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    
    # 4. Aplicar el filtro a la señal usando filtfilt (Filtro de fase cero para evitar desfases artificiales)
    y_filtrada = signal.filtfilt(b, a, y)
    
    return y_filtrada
def convertir_a_tensor_pytorch(matriz, agregar_canal=True):
    """
    [SEMANA 1 - ADELANTO IA]
    Convierte opcionalmente una matriz de características (Mel-Spectrogram o MFCC)
    en un torch.Tensor de PyTorch listo para la CNN.
    Si agregar_canal=True, transforma la forma de (features, frames) a (1, features, frames).
    """
    try:
        import torch
        print("🤖 Convirtiendo características a tensor de PyTorch...")
        tensor = torch.from_numpy(matriz).float()
        
        if agregar_canal:
            # Añade la dimensión del canal (Batch/Channel) requerida por las CNNs de PyTorch
            tensor = tensor.unsqueeze(0) 
            
        return tensor
    except ImportError:
        print("⚠️ Advertencia: PyTorch no está instalado en este entorno local.")
        print("   Se retornará la matriz original de NumPy para no romper el flujo.")
        return matriz


def guardar_espectrograma_limpio(mel_db, ruta_salida):
    """
    [SEMANA 2 - LUNES (ADELANTADO)]
    Toma la matriz mel_db y la guarda como una imagen PNG completamente limpia.
    Sin ejes, sin leyendas, sin bordes blancos ni márgenes. 
    Ideal para que la CNN la procese como imagen o para mostrar de forma estética en Streamlit.
    """
    print(f"🖼️ Guardando espectrograma limpio en: {ruta_salida}...")
    
    # Asegurar que el directorio de salida exista
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    # Configurar la figura dinámicamente según el tamaño de la matriz
    plt.figure(figsize=(10, 4))
    
    # Graficar sin ejes ni marcos
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Dibujar el espectrograma (usamos el mapa de color 'magma' o 'viridis' común en audio)
    librosa.display.specshow(mel_db, cmap='magma')
    
    # Guardar con resolución y removiendo cualquier remanente de espacio en blanco
    plt.savefig(ruta_salida, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close() # Liberar memoria de matplotlib
    print("✅ Imagen guardada exitosamente.")