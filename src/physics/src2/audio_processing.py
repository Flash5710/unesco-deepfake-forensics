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
def contrastar_espectrogramas_forenses(ruta_real, ruta_falso, target_sr=16000):
    """
    [SEMANA 2 - MARTES]
    Carga un audio de referencia (Real) y un audio sospechoso (Falso/Deepfake),
    extrae sus propiedades físicas espectrales, de energía y de fase, y calcula
    las métricas de contraste forense para la interfaz.
    """
    print("\n🔬 Iniciando análisis comparativo forense...")
    
    # 1. Cargar y normalizar ambas señales acústicas
    y_real, sr_real = cargar_y_normalizar_audio(ruta_real, target_sr=target_sr)
    y_falso, sr_falso = cargar_y_normalizar_audio(ruta_falso, target_sr=target_sr)
    
    # 2. Extracción de espectrogramas de Mel y MFCCs
    mel_db_real, mfccs_real = calcular_stft_y_mel(y_real, sr_real)
    mel_db_falso, mfccs_falso = calcular_stft_y_mel(y_falso, sr_falso)
    
    # 3. Análisis de Fase (Tu métrica matemática de regularidad)
    fase_real = calcular_regularidad_fase(y_real)
    fase_falso = calcular_regularidad_fase(y_falso)
    
    # 4. Análisis del Filtro Pasa-Altas (>4kHz) para detectar Vocoders
    y_altas_real = aplicar_filtro_paso_alto(y_real, sr_real, cutoff_freq=4000.0)
    y_altas_falso = aplicar_filtro_paso_alto(y_falso, sr_falso, cutoff_freq=4000.0)
    
    # Calcular ratios de energía espectral anómala
    energia_total_real = np.sum(y_real**2)
    energia_alta_real = np.sum(y_altas_real**2)
    ratio_alta_real = (energia_alta_real / energia_total_real) * 100 if energia_total_real > 0 else 0
    
    energia_total_falso = np.sum(y_falso**2)
    energia_alta_falso = np.sum(y_altas_falso**2)
    ratio_alta_falso = (energia_alta_falso / energia_total_falso) * 100 if energia_total_falso > 0 else 0
    
    # 5. Calcular métricas de distancia / contraste estadístico
    # Distancia Euclidiana promedio entre los coeficientes MFCC (mide disparidad tímbrica general)
    # Ajustamos las dimensiones temporales por si los audios no miden exactamente lo mismo
    min_frames = min(mfccs_real.shape[1], mfccs_falso.shape[1])
    distancia_mfcc = np.linalg.norm(mfccs_real[:, :min_frames] - mfccs_falso[:, :min_frames], axis=0)
    disparidad_timbrica_media = float(np.mean(distancia_mfcc))
    
    # Estructurar el reporte de contraste físico
    reporte_contraste = {
        "fase": {
            "real": float(fase_real),
            "falso": float(fase_falso),
            "diferencia_absoluta": float(abs(fase_real - fase_falso))
        },
        "energia_alta_frecuencia": {
            "real_porcentaje": float(ratio_alta_real),
            "falso_porcentaje": float(ratio_alta_falso),
            "desviacion_vocoder": float(abs(ratio_alta_real - ratio_alta_falso))
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
    [SEMANA 2 - MIÉRCOLES]
    Toma la matriz del espectrograma de Mel y genera un objeto de figura interactivo
    utilizando Plotly. Permite hacer zoom y examinar los valores en dB en tiempo real.
    Ideal para que el Matemático lo embeba en Streamlit usando st.plotly_chart().
    """
    try:
        import plotly.graph_objects as go
        print("📊 Diseñando mapa de calor interactivo con Plotly...")
        
        # 1. Crear los ejes físicos de tiempo y frecuencia de Mel
        times = librosa.times_like(mel_db, sr=sr, hop_length=hop_length)
        # Aproximación de frecuencias Mel para el eje Y
        mel_frequencies = librosa.mel_frequencies(n_mels=mel_db.shape[0], fmax=sr//2)
        
        # 2. Construir la superficie interactiva de calor (Heatmap)
        fig = go.Figure(data=go.Heatmap(
            z=mel_db,
            x=times,
            y=mel_frequencies,
            colorscale='Magma',
            zmin=-80, zmax=0, # Rango estándar de dB
            colorbar=dict(title="Potencia (dB)")
        ))
        
        # 3. Estilizar el layout forense para la UI modular
        fig.update_layout(
            title=dict(text="<b>Firma Espectral Interactiva (Auditoría Forense)</b>", font=dict(size=16)),
            xaxis=dict(title="Tiempo (segundos)"),
            yaxis=dict(title="Frecuencia Psicoacústica (Hz - Escala Mel)"),
            margin=dict(l=50, r=50, b=50, t=50),
            height=400,
            template="plotly_dark" # Fondo oscuro estético para el Hackaton
        )
        return fig
    except ImportError:
        print("⚠️ Advertencia: Plotly no está instalado en este entorno. Se omite el mapa interactivo.")
        return None
# Bloque de prueba local
if __name__ == "__main__":
    print("\n--- 🔬 PIPELINE REAL DE INGENIERÍA DE SEÑALES (SEMANAS 1 & 2) ---")
    
    # 1. Definición de rutas base de prueba para tu entorno local
    ruta_video_test = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\test_video.mp4"
    ruta_wav_real = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\temp_audio_real.wav"
    ruta_wav_falso = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\temp_audio_falso.wav"
    
    # Rutas para guardar las imágenes generadas por ti (Físico)
    ruta_img_clean = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\test_espectrograma_clean.png"
    ruta_img_forense = r"C:\Users\NW USER\OneDrive\Documentos\unesco-deepfake-forensics\src\physics\data\samples\test_espectrograma_forense.png"

    try:
        # === SEMANA 1: Cimientos del Sistema y Extracción ===
        print("\n=== STEP 1: Extracción y Preparación Básica (S1 - Martes) ===")
        extraer_audio_de_video(ruta_video_test, ruta_wav_real)
        
        # Simulación de audio falso duplicando la muestra para consistencia local
        if not os.path.exists(ruta_wav_falso):
            print("⚠️ Nota: No se detectó un archivo deepfake explícito, duplicando muestra para test de estabilidad.")
            import shutil
            shutil.copy(ruta_wav_real, ruta_wav_falso)

        print("\n=== STEP 2: Pruebas Unitarias de Espectrogramas y Tensores (S1 - Mié/Jue/Vie) ===")
        y, sr = cargar_y_normalizar_audio(ruta_wav_real, target_sr=16000)
        mel_db, mfccs = calcular_stft_y_mel(y, sr)
        
        # Validar los entregables para el Ingeniero de IA (Tensores) y Matplotlib (Imágenes)
        mel_tensor = convertir_a_tensor_pytorch(mel_db, agregar_canal=True)
        guardar_espectrograma_limpio(mel_db, ruta_img_clean)
        guardar_espectrograma_forense(mel_db, sr, ruta_salida=ruta_img_forense)
        
        print(f"  -> Dimensiones de matriz Mel de datos: {mel_db.shape}")
        if hasattr(mel_tensor, 'shape'):
            print(f"  -> Tensor listo para la CNN de IA: {list(mel_tensor.shape)}")

        # === SEMANA 2: Modelado Analítico y Construcción ===
        print("\n=== STEP 3: MÓDULO FORENSE DE CONTRASTE (S2 - Martes) ===")
        resultado = contrastar_espectrogramas_forenses(ruta_wav_real, ruta_wav_falso)
        
        print("\n📊 REPORTE DE MÉTRICAS COMPARATIVAS EXTRAÍDAS:")
        print(f"  -> Varianza de Fase [Audio Real]:   {resultado['fase']['real']:.6f}")
        print(f"  -> Varianza de Fase [Audio Falso]:  {resultado['fase']['falso']:.6f}")
        print(f"  -> Diferencia Delta de Fase:        {resultado['fase']['diferencia_absoluta']:.6f}")
        print("-" * 60)
        print(f"  -> Energía Espectral >4kHz [Real]:  {resultado['energia_alta_frecuencia']['real_porcentaje']:.4f}%")
        print(f"  -> Energía Espectral >4kHz [Falso]: {resultado['energia_alta_frecuencia']['falso_porcentaje']:.4f}%")
        print(f"  -> Desviación de Alta Frecuencia:   {resultado['energia_alta_frecuencia']['desviacion_vocoder']:.4f}%")
        print("-" * 60)
        print(f"  -> Disparidad Tímbrica Global MFCC: {resultado['disparidad_timbrica_mfcc']:.4f}")

        print("\n=== STEP 4: MAPA DE CALOR INTERACTIVO PARA LA UI (S2 - Miércoles) ===")
        # Mandamos el espectrograma del audio falso/analizado para generar el mapa interactivo
        fig_interactiva = generar_mapa_calor_interactivo(resultado['matrices']['mel_db_falso'], sr=sr)
        
        if fig_interactiva is not None:
            print("  ✅ Objeto Plotly generado exitosamente.")
            print("  👉 Tu Matemático podrá renderizarlo mañana usando: st.plotly_chart(fig_interactiva)")
        
        print("\n📦 CONEXIÓN DE DATOS COMPLETADA: Todos los entregables físicos del Miércoles están listos.")

    except Exception as e:
        print(f"\n❌ Error crítico en la ejecución del Pipeline: {str(e)}")
        print("Verifica que las rutas de los archivos .mp4/.wav existan o estén bien escritas en tu disco.")
