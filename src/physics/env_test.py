import librosa
import numpy as np
import scipy.signal as signal
import moviepy as mp 
import matplotlib.pyplot as plt

print("🚀 ¡Entorno configurado correctamente!")
print( 'Librerías verificadas:' )
print(f" - Librosa versión: {librosa.__version__}")
print(f" - NumPy versión: {np.__version__}")

# Simulación física rápida: Generar un tono puro de 1 kHz (Voz humana idealizada)
sr = 22050  # Tasa de muestreo estándar en audio experimental
duracion = 1.0  # 1 segundo
t = np.linspace(0, duracion, int(sr * duracion), endpoint=False)
frecuencia_fundamental = 1000.0  # 1 kHz
señal = np.sin(2 * np.pi * frecuencia_fundamental * t)

print(f"\nSeñal de prueba generada: {len(señal)} muestras a {sr}Hz.")