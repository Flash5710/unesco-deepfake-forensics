import os
import numpy as np
import librosa
import moviepy as mp
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa.display

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

def guardar_espectrograma_forense(mel_db, sr, hop_length=256, ruta_salida="data/samples/espectrograma_forense.png", interactivo=False):
    """
    [SEMANA 2 - LUNES]
    Genera y guarda un espectrograma de diagnóstico con ejes, etiquetas y escala de decibelios.
    Diseñado para el mapa de calor educativo que se mostrará en la interfaz de Streamlit.
    """
    print(f"📊 Generando espectrograma forense (educativo) en: {ruta_salida}...")
    
    # Asegurar que el directorio de salida exista
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    # Configurar dimensiones estéticas del gráfico
    plt.figure(figsize=(10, 4.5))
    
    # Graficar con el mapa de calor 'magma' (resalta anomalías espectrales del vocoder)
    img = librosa.display.specshow(
        mel_db, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='mel', 
        cmap='magma'
    )
    
    # Añadir elementos educativos para el usuario final del detector
    plt.colorbar(img, format='%+2.0f dB')
    plt.title("Análisis Espectral de Voz (Mapa Forense de Calor)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Tiempo (segundos)", fontsize=10)
    plt.ylabel("Frecuencia Psicoacústica (Escala Mel)", fontsize=10)
    
    # Ajustar para evitar recortes de texto en los bordes
    plt.tight_layout()
    
    # Guardar en disco
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print("✅ Espectrograma forense guardado exitosamente.")
    
def calcular_regularidad_fase(y, n_fft=1024, hop_length=256):
    """
    [SEMANA 1 - MIÉRCOLES]
    Calcula la métrica de regularidad de fase para analizar la continuidad 
    de la onda. Los vocoders de IA suelen dejar rupturas de fase artificiales.
    Devuelve la varianza de la derivada de la fase en el tiempo (métrica de inestabilidad).
    """
    print("📐 Analizando regularidad de la fase de la onda...")
    
    # 1. Calcular la STFT compleja (sin extraer la magnitud absoluta)
    stft_compleja = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # 2. Extraer el ángulo (fase) en radianes
    fase = np.angle(stft_compleja)
    
    # 3. Calcular la derivada de la fase a lo largo del tiempo (diferencias finitas)
    delta_fase = np.diff(fase, axis=1)
    
    # 4. Envolver la fase entre [-pi, pi] para evitar saltos artificiales de 2*pi
    delta_fase_envuelta = np.angle(np.exp(1j * delta_fase))
    
    # 5. La varianza temporal nos dice qué tan "caótica" o discontinua es la fase.
    # Un valor anormalmente bajo o alto en ciertas bandas delata un deepfake.
    varianza_fase = np.var(delta_fase_envuelta, axis=1)
    
    # Retornamos el promedio global de la inestabilidad de fase como métrica base
    métrica_estabilidad = np.mean(varianza_fase)
    
    return métrica_estabilidad

# Bloque de prueba local
if __name__ == "__main__":
    print("\n--- 🔬 PIPELINE REAL DE INGENIERÍA DE SEÑALES (SEMANAS 1 & 2) ---")
    
    # 1. Definir rutas absolutas/relativas correctas en tu entorno de trabajo
    ruta_video_real = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\test_video.mp4"
    ruta_wav_temporal = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\temp_audio.wav"
    
    try:
        # 2. Extracción física del audio desde el contenedor MP4[cite: 3]
        extraer_audio_de_video(ruta_video_real, ruta_wav_temporal) 
        
        # 3. Carga y normalización de la señal (Remuestreo a 16kHz para Nyquist óptimo)[cite: 3]
        sr_objetivo = 16000
        y, sr = cargar_y_normalizar_audio(ruta_wav_temporal, target_sr=sr_objetivo)
        
        print(f"📊 Información física del audio real:")
        print(f"  -> Duración: {len(y)/sr:.2f} segundos")
        print(f"  -> Total de muestras analizadas: {len(y)}")
        
        # 4. PRODUCTO PARA EL INGENIERO DE IA: Espectrograma de Mel Completo[cite: 3]
        mel_db_completo, mfccs = calcular_stft_y_mel(y, sr) 
        
        # 5. PRODUCTO PARA EL MATEMÁTICO: Aislamiento de Altas Frecuencias (>4kHz)[cite: 3]
        y_altas_frecuencias = aplicar_filtro_paso_alto(y, sr, cutoff_freq=4000.0) 
        mel_db_filtrado, _ = calcular_stft_y_mel(y_altas_frecuencias, sr)
        
        # PASO 5.5 - CÁLCULO DE MÉTRICA DE FASE PARA EL MATEMÁTICO (Semana 1 - Miércoles)[cite: 3, 4]
        inestabilidad_fase = calcular_regularidad_fase(y, n_fft=1024, hop_length=256) 
        
        # 1. Convertir el espectrograma Mel a Tensor para el Ingeniero de IA (Semana 1 - Adelanto)[cite: 3]
        mel_tensor = convertir_a_tensor_pytorch(mel_db_completo, agregar_canal=True)
        
        # 2. Guardar espectrograma limpio en disco (Semana 2 - Lunes - Máquina)[cite: 3, 4]
        ruta_imagen_test = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\test_espectrograma_clean.png"
        guardar_espectrograma_limpio(mel_db_completo, ruta_imagen_test) 
        
        # === NUEVO: ENTREGABLE VISUAL FORENSE (Semana 2 - Lunes - Humano) ===[cite: 4]
        ruta_imagen_forense = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\test_espectrograma_forense.png"
        guardar_espectrograma_forense(mel_db_completo, sr=sr, ruta_salida=ruta_imagen_forense)
        # ====================================================================
        
        # 6. Extracción de métricas de energía para los modelos estadísticos[cite: 3]
        energia_total = np.sum(y**2)
        energia_alta = np.sum(y_altas_frecuencias**2)
        ratio_anomalia = (energia_alta / energia_total) * 100 if energia_total > 0 else 0
        
        print("\n📦 CONEXIÓN DE DATOS COMPLETADA CON ÉXITO:")
        print(f"  ✅ Matriz enviada a la CNN (IA): {mel_db_completo.shape}")
        
        # Muestra la forma del tensor con canales solo si PyTorch está disponible[cite: 3]
        if hasattr(mel_tensor, 'shape'):
            print(f"  🤖 Tensor de PyTorch generado para la Dataset de IA: {list(mel_tensor.shape)}")
        
        print(f"  ✅ Coeficientes enviados a Métricas (Matemático): {mfccs.shape}") 
        print(f"  ✅ Métrica de Inestabilidad de Fase: {inestabilidad_fase:.6f}")
        print(f"  ✅ Análisis Físico Forense: {ratio_anomalia:.4f}% de la energía en altas frecuencias (>4kHz).")
        
    except Exception as e:
        print(f"\n❌ Error en el procesamiento del archivo real: {str(e)}")
        print("Asegúrate de que el archivo 'test_video.mp4' exista en la ruta indicada y no esté corrupto.")
