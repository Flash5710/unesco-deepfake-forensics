import os
import time
import warnings
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')  # Evita que se abran ventanas flotantes de la GUI en servidores/Streamlit
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==============================================================================
# CONSTANTES GLOBALES DE PROCESAMIENTO DE AUDIO
# ==============================================================================
N_FFT = 1024
HOP_LENGTH = 256
TARGET_SR = 16000
N_MELS = 128
N_MFCC = 13
CUTOFF_FREQ_HZ = 4000.0


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
    audio.write_audiofile(ruta_salida_audio, fps=TARGET_SR, nbytes=2, codec='pcm_s16le', logger=None)
    
    audio.close()
    video.close()   
    
    print(f"✅ Audio extraído exitosamente en: {ruta_salida_audio}")
    return ruta_salida_audio


def cargar_y_normalizar_audio(ruta_audio, target_sr=TARGET_SR):
    """
    Carga un archivo de audio usando librosa, fuerza la monofonización
    y normaliza la amplitud pico a 1.0 para mantener la consistencia estadística.
    Maneja adecuadamente señales de silencio total para evitar división por cero.
    """
    print(f"💿 Cargando y preprocesando señal acústica a {target_sr} Hz...")
    y, sr = librosa.load(ruta_audio, sr=target_sr)
    
    if len(y) > 0:
        max_amp = np.max(np.abs(y))
        # Guard contra división por cero si el audio es silencio absoluto
        if max_amp > 0:
            y = y / max_amp
        
    return y, sr


# TODO: envolver con @st.cache_data en app.py
def calcular_stft_y_mel(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, stft_compleja=None):
    """
    Calcula o recibe la Transformada de Fourier de Tiempo Reducido (STFT)
    y la transforma a la Escala de Mel para análisis psicoacústico.
    Retorna la matriz de espectrograma Mel en dB, coeficientes MFCC y la STFT compleja.
    """
    print("🔮 Calculando STFT y Espectrograma de Mel...")
    if stft_compleja is None:
        stft_compleja = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    stft_magnitud = np.abs(stft_compleja)
    
    espectrograma_mel = librosa.feature.melspectrogram(
        S=stft_magnitud**2, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=N_MELS
    )
    
    mel_db = librosa.power_to_db(espectrograma_mel, ref=np.max)
    mfccs = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=N_MFCC)
    
    return mel_db, mfccs, stft_compleja


def aplicar_filtro_paso_alto(y, sr, cutoff_freq=CUTOFF_FREQ_HZ, order=5):
    """
    Aplica un filtro digital Butterworth de tipo paso-alto para aislar
    las frecuencias superiores a 'cutoff_freq' (por defecto 4kHz).
    
    Nota técnica: scipy.signal.filtfilt aplica el filtro en dirección directa y reversa
    (doble pasada) para garantizar fase cero. Por ende, el orden efectivo del filtrado 
    es aproximadamente el doble del parámetro 'order' proporcionado (ej. order=5 -> orden efectivo 10).
    """
    print(f"🎛️ Aplicando filtro paso-alto Butterworth (Corte: {cutoff_freq} Hz, Orden nominal: {order})...")
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y_filtrada = signal.filtfilt(b, a, y)
    
    return y_filtrada


# TODO: envolver con @st.cache_data en app.py
def calcular_regularidad_fase(y=None, n_fft=N_FFT, hop_length=HOP_LENGTH, stft_compleja=None):
    """
    Calcula la métrica de regularidad de fase para analizar la continuidad de la onda.
    Acepta opcionalmente una 'stft_compleja' precomputada para evitar re-cálculos redundantes.
    Devuelve la varianza de la derivada de la fase en el tiempo (métrica de inestabilidad).
    """
    print("📐 Analizando regularidad de la fase de la onda...")
    if stft_compleja is None:
        if y is None:
            raise ValueError("Se debe proporcionar al menos la señal temporal 'y' o la 'stft_compleja'.")
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


def guardar_espectrograma_forense(mel_db, sr, hop_length=HOP_LENGTH, ruta_salida="data/samples/espectrograma_forense.png"):
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


def contrastar_espectrogramas_forenses(ruta_real, ruta_falso, target_sr=TARGET_SR):
    """
    Compara propiedades físicas espectrales, de energía y de fase entre muestras reales y sospechosas.
    Reutiliza la STFT precomputada para optimizar el flujo comparativo.
    """
    print("\n🔬 Iniciando análisis comparativo forense...")
    y_real, sr_real = cargar_y_normalizar_audio(ruta_real, target_sr=target_sr)
    y_falso, sr_falso = cargar_y_normalizar_audio(ruta_falso, target_sr=target_sr)
    
    mel_db_real, mfccs_real, stft_real = calcular_stft_y_mel(y_real, sr_real)
    mel_db_falso, mfccs_falso, stft_falso = calcular_stft_y_mel(y_falso, sr_falso)
    
    fase_real = calcular_regularidad_fase(stft_compleja=stft_real)
    fase_falso = calcular_regularidad_fase(stft_compleja=stft_falso)
    
    y_altas_real = aplicar_filtro_paso_alto(y_real, sr_real, cutoff_freq=CUTOFF_FREQ_HZ)
    y_altas_falso = aplicar_filtro_paso_alto(y_falso, sr_falso, cutoff_freq=CUTOFF_FREQ_HZ)
    
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


# TODO: envolver con @st.cache_data en app.py
def generar_mapa_calor_interactivo(
    mel_db, 
    sr, 
    y_audio=None, 
    stft_compleja=None, 
    hop_length=HOP_LENGTH, 
    n_fft=N_FFT, 
    umbral_anomalia_db=-30.0, 
    freq_corte_hz=CUTOFF_FREQ_HZ, 
    top_n_anomalias=2
):
    """
    [SEMANA 2 - MIÉRCOLES & SEMANA 3 - MARTES]
    Genera un Heatmap interactivo en Plotly con detección y resaltado de zonas sospechosas.
    Reutiliza la 'stft_compleja' si es provista para evitar recalcularla en runtime.
    
    Retorna:
    - fig: Objeto Figure de Plotly
    - warning_msg: Mensaje de advertencia para la UI de Streamlit (o None si todo está OK)
    """
    warning_msg = None
    
    try:
        import plotly.graph_objects as go
        print("📊 Diseñando mapa de calor forense con validación de fase vs. sibilantes...")
        
        times = librosa.times_like(mel_db, sr=sr, hop_length=hop_length)
        mel_frequencies = librosa.mel_frequencies(n_mels=mel_db.shape[0], fmax=sr//2)
        
        # 1. Figura base del Espectrograma Mel
        fig = go.Figure(data=go.Heatmap(
            z=mel_db, 
            x=times, 
            y=mel_frequencies,
            colorscale='Inferno', 
            zmin=-80, 
            zmax=0,
            colorbar=dict(
                title=dict(text="Potencia (dB)", font=dict(color="#FFFFFF", size=11)),
                tickfont=dict(color="#D0D0D0", size=9),
                thickness=15,
                len=0.9
            ),
            hovertemplate="<b>Tiempo:</b> %{x:.2f} s<br>" +
                          "<b>Frecuencia:</b> %{y:.0f} Hz<br>" +
                          "<b>Potencia:</b> %{z:.1f} dB<extra></extra>"
        ))
        
        # 2. Análisis Multidominio (Energía + Fase)
        idx_altas_freq = np.where(mel_frequencies >= freq_corte_hz)[0]
        
        if len(idx_altas_freq) > 0:
            submatriz_altas = mel_db[idx_altas_freq, :]
            mask_energia = submatriz_altas > umbral_anomalia_db
            
            stft_eval = stft_compleja
            
            # Fallback a y_audio usando n_fft parametrizado y detección de desajustes
            if stft_eval is None and y_audio is not None and len(y_audio) > 0:
                stft_eval = librosa.stft(y_audio, n_fft=n_fft, hop_length=hop_length)
                n_frames_esperados = mel_db.shape[1]
                if stft_eval.shape[1] != n_frames_esperados:
                    warning_msg = (
                        f"⚠️ Desajuste de frames: La STFT calculada produce {stft_eval.shape[1]} frames, "
                        f"mientras que mel_db tiene {n_frames_esperados}. Transmita 'stft_compleja' para mayor precisión."
                    )
                    warnings.warn(warning_msg, UserWarning)
            
            # Warning para UI/Console en modo degraded
            if stft_eval is None:
                warning_msg = "⚠️ Modo Degradado: Operando en 'Solo Energía'. Falsos positivos por sibilantes (/s/, /f/) son posibles."
                print(f"⚠️ LOG FORENSE: {warning_msg}")
                mask_sospechosa = mask_energia
            else:
                # Aislamiento de la banda >4kHz usando n_fft parametrizado
                stft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                idx_stft_altas = np.where(stft_freqs >= freq_corte_hz)[0]
                
                if len(idx_stft_altas) > 0:
                    fase_banda_alta = np.angle(stft_eval[idx_stft_altas, :])
                else:
                    fase_banda_alta = np.angle(stft_eval)
                
                delta_fase = np.diff(fase_banda_alta, axis=1)
                delta_fase_envuelta = np.angle(np.exp(1j * delta_fase))
                varianza_fase_frame = np.var(delta_fase_envuelta, axis=0)
                
                # Edge Padding
                if len(varianza_fase_frame) < mel_db.shape[1]:
                    diferencia_frames = mel_db.shape[1] - len(varianza_fase_frame)
                    varianza_fase_frame = np.pad(varianza_fase_frame, (0, diferencia_frames), mode='edge')
                elif len(varianza_fase_frame) > mel_db.shape[1]:
                    varianza_fase_frame = varianza_fase_frame[:mel_db.shape[1]]
                
                umbral_fase = np.percentile(varianza_fase_frame, 65)
                mask_fase_inestable = varianza_fase_frame > umbral_fase
                
                mask_sospechosa = mask_energia & mask_fase_inestable[np.newaxis, :]

            if np.any(mask_sospechosa):
                z_anomalias = np.full_like(mel_db, np.nan)
                
                for i_rel, i_abs in enumerate(idx_altas_freq):
                    z_anomalias[i_abs, mask_sospechosa[i_rel, :]] = mel_db[i_abs, mask_sospechosa[i_rel, :]]
                
                fig.add_trace(go.Heatmap(
                    z=z_anomalias,
                    x=times,
                    y=mel_frequencies,
                    colorscale=[[0, 'rgba(255,50,50,0.7)'], [1, 'rgba(255,255,0,1)']],
                    showscale=False,
                    hoverinfo='skip'
                ))
                
                # Top-N Anotaciones
                indices_validos = np.argwhere(~np.isnan(z_anomalias))
                if len(indices_validos) > 0:
                    valores_anomalias = [z_anomalias[f, t] for f, t in indices_validos]
                    ordenados = np.argsort(valores_anomalias)[::-1]
                    picos_seleccionados = []
                    tiempos_usados = []
                    
                    for idx in ordenados:
                        f_idx, t_idx = indices_validos[idx]
                        t_val = times[t_idx]
                        
                        if all(abs(t_val - tu) > 0.4 for tu in tiempos_usados):
                            picos_seleccionados.append((f_idx, t_idx))
                            tiempos_usados.append(t_val)
                            if len(picos_seleccionados) >= top_n_anomalias:
                                break

                    for k, (f_idx, t_idx) in enumerate(picos_seleccionados):
                        tiempo_pico = times[t_idx]
                        freq_pico = mel_frequencies[f_idx]
                        
                        fig.add_annotation(
                            x=tiempo_pico,
                            y=freq_pico,
                            text=f"⚠️ Artefacto Vocoder #{k+1}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="#FF3333",
                            arrowsize=1.2,
                            arrowwidth=2,
                            ax=0 if k % 2 == 0 else 20,
                            ay=-35 - (k * 15),
                            font=dict(size=10, color="#FFFFFF"),
                            bgcolor="rgba(180, 0, 0, 0.8)",
                            bordercolor="#FF3333",
                            borderwidth=1
                        )

        fig.update_layout(
            title=dict(
                text="<b>Análisis Forense: Zonas de Alta Incoherencia de Fase y Energía</b>", 
                font=dict(size=15, color="#FFFFFF"),
                x=0.01, y=0.95
            ),
            xaxis=dict(
                title=dict(text="Tiempo (segundos)", font=dict(color="#E0E0E0", size=11)),
                tickfont=dict(color="#B0B0B0", size=9),
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text="Frecuencia (Hz - Escala Mel)", font=dict(color="#E0E0E0", size=11)),
                tickfont=dict(color="#B0B0B0", size=9),
                showgrid=False,
                zeroline=False
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=50, r=20, t=45, b=45),
            height=400,
            autosize=True
        )
        return fig, warning_msg
        
    except ImportError:
        print("⚠️ Plotly no instalado. Se omite el mapa interactivo.")
        return None, "Plotly no disponible en el sistema."


def generar_espectrograma_forense_web(mel_db, sr, hop_length=HOP_LENGTH, titulo="Análisis Espectral Forense de Voz"):
    """
    [SEMANA 2 - JUEVES & SEMANA 3 - MARTES]
    Genera un objeto Figure de Matplotlib con paletas perceptualmente uniformes 
    y estéticas oscuras de alto contraste para renderizado directo en Streamlit con st.pyplot().
    """
    print("📊 Generando objeto Matplotlib optimizado con paleta de alto contraste...")
    
    # Aplicar un estilo oscuro consistente con la app
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 4.2), facecolor='none') # Fondo transparente
        ax.set_facecolor('none')
        
        # 'inferno' o 'magma' garantizan legibilidad en pantallas de alta resolución
        img = librosa.display.specshow(
            mel_db, 
            sr=sr, 
            hop_length=hop_length, 
            x_axis='time', 
            y_axis='mel', 
            cmap='inferno', 
            ax=ax
        )
        
        # Configuración de barra de colores y etiquetas
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.tick_params(labelsize=8, colors='#D0D0D0')
        cbar.set_label("Intensidad (dB)", color='#FFFFFF', fontsize=9)
        
        ax.set_title(titulo, fontsize=12, fontweight='bold', color='#FFFFFF', pad=12)
        ax.set_xlabel("Tiempo (segundos)", fontsize=10, color='#E0E0E0')
        ax.set_ylabel("Frecuencia (Escala Mel)", fontsize=10, color='#E0E0E0')
        ax.tick_params(colors='#B0B0B0', labelsize=8)
        
        fig.tight_layout()
        return fig


def evaluar_latencia_procesamiento(ruta_video_test, target_sr=TARGET_SR):
    """
    [SEMANA 3 - MIÉRCOLES]
    Mide el tiempo de ejecución de cada etapa del pipeline del Físico
    para reportar latencias y asegurar que la UI mantenga un rendimiento fluido.
    Demuestra la optimización reutilizando la STFT a lo largo del pipeline.
    """
    print("\n⏱️ --- INICIANDO BENCHMARK DE LATENCIAS Y TIEMPOS DE CÓMPUTO ---")
    
    reporte_latencias = {}
    tiempo_total_inicio = time.perf_counter()
    
    # 1. Extracción de audio desde video
    t0 = time.perf_counter()
    ruta_wav = extraer_audio_de_video(ruta_video_test)
    reporte_latencias["1_extraccion_audio_wav"] = round(time.perf_counter() - t0, 4)
    
    # 2. Carga y Normalización
    t0 = time.perf_counter()
    y, sr = cargar_y_normalizar_audio(ruta_wav, target_sr=target_sr)
    reporte_latencias["2_carga_y_normalizacion"] = round(time.perf_counter() - t0, 4)
    
    duracion_audio_sec = len(y) / sr if sr > 0 else 0
    
    # 3. Cálculo Unificado de STFT, Mel-Spectrogram y MFCCs
    t0 = time.perf_counter()
    mel_db, mfccs, stft_compleja = calcular_stft_y_mel(y, sr)
    reporte_latencias["3_stft_mel_mfcc"] = round(time.perf_counter() - t0, 4)
    
    # 4. Filtro Paso-Alto (Butterworth)
    t0 = time.perf_counter()
    y_altas = aplicar_filtro_paso_alto(y, sr)
    reporte_latencias["4_filtro_paso_alto"] = round(time.perf_counter() - t0, 4)
    
    # 5. Regularidad de Fase (Reutilizando STFT)
    t0 = time.perf_counter()
    fase_var = calcular_regularidad_fase(stft_compleja=stft_compleja)
    reporte_latencias["5_regularidad_fase"] = round(time.perf_counter() - t0, 4)
    
    # 6. Generación de Mapa de Calor Interactivo (Plotly, reutilizando STFT)
    t0 = time.perf_counter()
    fig_plotly, _ = generar_mapa_calor_interactivo(mel_db, sr, y_audio=y, stft_compleja=stft_compleja)
    reporte_latencias["6_render_plotly_interactivo"] = round(time.perf_counter() - t0, 4)
    
    # 7. Generación de Espectrograma Web (Matplotlib)
    t0 = time.perf_counter()
    fig_matplotlib = generar_espectrograma_forense_web(mel_db, sr)
    reporte_latencias["7_render_matplotlib_web"] = round(time.perf_counter() - t0, 4)
    
    tiempo_total = round(time.perf_counter() - tiempo_total_inicio, 4)
    reporte_latencias["tiempo_total_pipeline"] = tiempo_total
    
    # Factor de Latencia en Tiempo Real (RTF = Tiempo Procesamiento / Duración Audio)
    rtf = round(tiempo_total / duracion_audio_sec, 3) if duracion_audio_sec > 0 else 0
    reporte_latencias["real_time_factor_rtf"] = rtf

    # --- IMPRESIÓN DEL REPORTE PARA EL EQUIPO ---
    print("\n📊 --- REPORTE DE LATENCIA FORENSE ---")
    print(f"🔊 Duración del Audio de Prueba: {duracion_audio_sec:.2f} segundos")
    print(f"⏱️ Tiempo Total de Cómputo:      {tiempo_total:.4f} segundos")
    print(f"⚡ Factor Real-Time (RTF):        {rtf}x {'(¡Más rápido que tiempo real!)' if rtf < 1.0 else '(Más lento que tiempo real)'}")
    print("-" * 50)
    for etapa, t in reporte_latencias.items():
        if "total" not in etapa and "rtf" not in etapa:
            porcentaje = round((t / tiempo_total) * 100, 1) if tiempo_total > 0 else 0
            print(f"  • {etapa.replace('_', ' ').capitalize()}: {t:.4f}s ({porcentaje}%)")
    print("-" * 50)
    
    return reporte_latencias


# --- BLOQUE DE PRUEBA LOCAL Y BENCHMARK DE LATENCIAS ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 PIPELINE LOCAL DE INGENIERÍA DE SEÑALES & BENCHMARK FORENSE")
    print("=" * 60)
    
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
        print(f"\n⚠️ No se encontró video de prueba en: {ruta_video_test}")
        print("💡 Coloca un archivo '.mp4' con ese nombre en dicha carpeta para ejecutar el benchmark completo.")
    else:
        try:
            # 1. Ejecución del pipeline modular tradicional
            print("\n--- 1. Ejecución de Prueba de Funcionalidades ---")
            extraer_audio_de_video(ruta_video_test, ruta_wav_real)
            
            if not os.path.exists(ruta_wav_falso):
                import shutil
                shutil.copy(ruta_wav_real, ruta_wav_falso)

            y, sr = cargar_y_normalizar_audio(ruta_wav_real, target_sr=TARGET_SR)
            mel_db, mfccs, stft_comp = calcular_stft_y_mel(y, sr)
            
            mel_tensor = convertir_a_tensor_pytorch(mel_db)
            guardar_espectrograma_limpio(mel_db, ruta_img_clean)
            guardar_espectrograma_forense(mel_db, sr, ruta_salida=ruta_img_forense)
            
            resultado_contraste = contrastar_espectrogramas_forenses(ruta_wav_real, ruta_wav_falso)
            print("✅ Prueba de funciones individuales completada con éxito.")

            # 2. Evaluación de Latencia y Tiempo de Cómputo (Tarea del Miércoles, Semana 3)
            print("\n--- 2. Benchmark de Tiempo de Cómputo (Miércoles, Semana 3) ---")
            reporte_latencias = evaluar_latencia_procesamiento(ruta_video_test)
            
            print("\n✅ Test de Pipeline Modularizado y Benchmark Finalizado Correctamente.")
            
        except Exception as e:
            print(f"\n❌ Error durante la ejecución del main local: {e}")
