import numpy as np

from src.math_core.normalization import cargar_y_normalizar_audio
from src.math_core.metrics import calcular_regularidad_fase
from src.physics.src2.audio_processing import aplicar_filtro_paso_alto

# NOTA: Simularemos el paso del filtro del Físico para validar dimensiones
def simulador_filtro_fisico(y):
    """Simula la salida del filtro manteniendo las dimensiones de entrada"""
    return y.copy()

def ejecutar_prueba():
    print("--- INICIANDO PRUEBA DE INTEGRACIÓN MATEMÁTICA ---")
    
    # 1. Creamos una señal de audio simulada (ruido blanco) de 1 segundo a 16kHz
    sr = 16000
    y_simulada = np.random.uniform(-1, 1, sr)
    print(f"✅ Señal cruda generada. Dimensiones iniciales (Vector): {y_simulada.shape}")
    
    # 2. Prueba del Físico
    y_filtrada = aplicar_filtro_paso_alto(y_simulada, sr=16000)
    
    if y_simulada.shape == y_filtrada.shape:
        print(f"✅ VALIDACIÓN VIERNES SUPERADA (FILTRO REAL): El filtro mantiene la integridad dimensional {y_filtrada.shape}.")
    else:
        print("❌ ERROR: El filtro físico destruyó la dimensión del vector.")
        
    # 3. Prueba de Normalización
    y_normalizada = y_filtrada / np.max(np.abs(y_filtrada))
    print(f"✅ Normalización matemática aplicada. Rango actual: [{np.min(y_normalizada):.2f}, {np.max(y_normalizada):.2f}]")
    
    # 4. Prueba de la Métrica de Fase
    try:
        inestabilidad = calcular_regularidad_fase(y_normalizada)
        print(f"✅ VALIDACIÓN SUPERADA: Métrica calculada sin errores de división. Resultado: {inestabilidad:.6f}")
    except Exception as e:
        print(f"❌ ERROR en el módulo de fase: {e}")

if __name__ == "__main__":
    ejecutar_prueba()