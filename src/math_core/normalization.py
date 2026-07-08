import numpy as np
import librosa

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