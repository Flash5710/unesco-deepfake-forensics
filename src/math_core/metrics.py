import numpy as np
import librosa

def calcular_regularidad_fase(y, n_fft=2048, hop_length=512):
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