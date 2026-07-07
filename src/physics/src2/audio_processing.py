import os
import numpy as np
import librosa
import moviepy as mp
import scipy.signal as signal

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

def cargar_y_normalizar_audio(ruta_audio, target_sr=16000):
    """
    Carga un archivo de audio usando librosa, fuerza la monofonización
    y normaliza la amplitud pico a 1.0 para mantener la consistencia estadística.
    """
    print(f"💿 Cargando y preprocesando señal acústica a {target_sr} Hz...")
    
    # librosa.load convierte automáticamente a mono por defecto (mono=True)
    y, sr = librosa.load(ruta_audio, sr=target_sr)
    
    # Normalización de amplitud (Evita fluctuaciones por volumen de grabación)
    if len(y) > 0:
        y = y / np.max(np.abs(y))
        
    return y, sr
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
    print(f"🎛️ Aplicando filtro paso-alto Butterworth (Corte: {cutoff_freq} Hz, Orden: {order})...")
    
    # 1. Calcular la frecuencia de Nyquist (Límite físico del sistema analizado)
    nyquist = 0.5 * sr
    
    # 2. Normalizar la frecuencia de corte según Nyquist (Requisito de Scipy)
    normal_cutoff = cutoff_freq / nyquist
    
    # 3. Diseñar el filtro (b y a son los coeficientes de la función de transferencia)
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    
    # 4. Aplicar el filtro a la señal usando filtfilt (Filtro de fase cero para evitar desfases artificiales)
    y_filtrada = signal.filtfilt(b, a, y)
    
    return y_filtrada
# Bloque de prueba local
if __name__ == "__main__":
    print("\n--- 🔬 PIPELINE REAL DE INGENIERÍA DE SEÑALES (SEMANA 1) ---")
    
    # 1. Definir la ruta del video que nos diste
    ruta_video_real = r"C:\Users\NW USER\OneDrive\Escritorio\Roadto\hackaton_deepfake\data\samples\test_video.mp4"
    ruta_wav_temporal = "data/samples/temp_audio.wav"
    
    try:
        # 2. Extracción física del audio desde el contenedor MP4
        extraer_audio_de_video(ruta_video_real, ruta_wav_temporal)
        
        # 3. Carga y normalización de la señal (Remuestreo a 16kHz para Nyquist óptimo)
        sr_objetivo = 16000
        y, sr = cargar_y_normalizar_audio(ruta_wav_temporal, target_sr=sr_objetivo)
        
        print(f"📊 Información física del audio real:")
        print(f"  -> Duración: {len(y)/sr:.2f} segundos")
        print(f"  -> Total de muestras analizadas: {len(y)}")
        
        # 4. PRODUCTO PARA EL INGENIERO DE IA: Espectrograma de Mel Completo (Sin alterar)
        # La CNN necesita el espectro completo para aprender contextos armónicos reales
        mel_db_completo, mfccs = calcular_stft_y_mel(y, sr)
        
        # 5. PRODUCTO PARA EL MATEMÁTICO: Aislamiento de Altas Frecuencias (>4kHz)
        # Filtramos en paralelo para analizar los artefactos matemáticos del vocoder
        y_altas_frecuencias = aplicar_filtro_paso_alto(y, sr, cutoff_freq=4000.0)
        mel_db_filtrado, _ = calcular_stft_y_mel(y_altas_frecuencias, sr)
        
        # 6. Extracción de métricas de energía para los modelos estadísticos
        energia_total = np.sum(y**2)
        energia_alta = np.sum(y_altas_frecuencias**2)
        ratio_anomalia = (energia_alta / energia_total) * 100 if energia_total > 0 else 0
        
        print("\n📦 CONEXIÓN DE DATOS COMPLETADA CON ÉXITO:")
        print(f"  ✅ Matriz enviada a la CNN (IA): {mel_db_completo.shape} (Formato tensor listo)")
        print(f"  ✅ Coeficientes enviados a Métricas (Matemático): {mfccs.shape}")
        print(f"  ✅ Análisis Físico Forense: {ratio_anomalia:.4f}% de la energía total se concentra en la zona de artefactos (>4kHz).")
        
    except Exception as e:
        print(f"\n❌ Error en el procesamiento del archivo real: {str(e)}")
        print("Asegúrate de que el archivo 'test_video.mp4' exista en la ruta indicada y no esté corrupto.")
