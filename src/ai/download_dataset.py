import os
import numpy as np
import scipy.io.wavfile as wav

def generate_mock_data():
    base_dir = "./data"
    subdirs = [
        "train/real", "train/fake",
        "val/real", "val/fake"
    ]
    
    print("Generando entorno de datos local simulado...")
    
    # Crear cada directorio si no existe
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        
        # Generar 5 archivos de muestra por carpeta para el MVP de la Semana 1
        for i in range(5):
            file_name = f"sample_{i}.wav"
            file_path = os.path.join(path, file_name)
            
            # Crear una señal de audio simple (frecuencia de muestreo 16000Hz, 1 segundo)
            sample_rate = 16000
            t = np.linspace(0, 1.0, sample_rate, endpoint=False)
            
            # Las voces reales tienen armónicos naturales; las fakes simulan ruido de vocoder
            if "real" in subdir:
                signal = np.sin(2 * np.pi * 440 * t) # Tono puro 440Hz
            else:
                signal = np.random.normal(0, 0.5, sample_rate) # Ruido artificial
                
            # Normalizar a enteros de 16 bits para simular un archivo .wav real
            signal_int = np.int16(signal * 32767)
            wav.write(file_path, sample_rate, signal_int)
            
    print(" ¡Estructura /data completada con éxito con muestras locales!")
    print(" El pipeline de la Semana 1 está completamente listo para recibir el código de señales.")

if __name__ == "__main__":
    generate_mock_data()