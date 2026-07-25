import time
import torch
import numpy as np
from src.ai.predict import predecir_etiqueta

def ejecutar_prueba_estres(num_muestras=50):
    print(f" Iniciando prueba de estrés del modelo de IA con {num_muestras} ejecuciones continuas...")
    
    latencias = []
    errores = 0
    
    # Simulación de ráfaga de espectrogramas de audios reales/fake con variaciones de duración
    print(" Procesando lote intensivo de espectrogramas...")
    start_total = time.time()
    
    for i in range(num_muestras):
        try:
            # Simular variaciones temporales aleatorias (entre 30 y 100 frames)
            duracion_random = np.random.randint(30, 100)
            espectrograma_dummy = torch.randn(1, 128, duracion_random)
            
            t_inicio = time.time()
            etiqueta, confianza = predecir_etiqueta(espectrograma_dummy)
            t_fin = time.time()
            
            latencia_ms = (t_fin - t_inicio) * 1000
            latencias.append(latencia_ms)
            
        except Exception as e:
            errores += 1
            print(f" Error en la iteración {i+1}: {e}")

    tiempo_total = time.time() - start_total
    
    # Métricas de rendimiento y estrés
    latencia_promedio = np.mean(latencias)
    latencia_max = np.max(latencias)
    throughput = num_muestras / tiempo_total

    print("\n --- RESULTADOS DE LA PRUEBA DE ESTRÉS ---")
    print(f" Muestras procesadas exitosamente: {num_muestras - errores}/{num_muestras}")
    print(f" Tasa de procesamiento (Throughput): {throughput:.2f} audios/segundo")
    print(f" Latencia promedio por audio: {latencia_promedio:.2f} ms")
    print(f" Latencia máxima registrada: {latencia_max:.2f} ms")
    print(f" Tiempo total de prueba: {tiempo_total:.2f} segundos")
    
    if errores == 0:
        print(" ¡El modelo superó la prueba de estrés sin fallos de estabilidad!")
    else:
        print(f" Se detectaron {errores} fallos durante la prueba.")

if __name__ == "__main__":
    ejecutar_prueba_estres(num_muestras=30)