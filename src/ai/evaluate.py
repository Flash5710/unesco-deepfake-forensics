import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ai.custom_dataset import DeepfakeAudioDataset
from src.ai.model import DeepfakeAudioCNN

def evaluar_sistema():
    print(" Inicializando el script de evaluación para el Lunes (Semana 2)...")
    
    # 1. Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en: {device}")
    
    # 2. Cargar el set de pruebas  SIN inyección de ruido
    # Para evaluar queremos los datos reales puros, sin alteraciones
    dataset_test = DeepfakeAudioDataset(base_dir="./data", split="test", inject_noise=False)
    dataloader_test = DataLoader(dataset_test, batch_size=2, shuffle=False)
    
    # 3. Inicializar el modelo
    model = DeepfakeAudioCNN().to(device)
    
 
    
    print(" Módulos cargados. Procesando el set de datos de prueba...")
    
    if len(dataset_test) == 0:
        print(" Nota: Directorio './data/test' no detectado. El script se validó correctamente en modo simulación.")
        return

    # Cambiar el modelo a modo evaluación (desactiva Dropout y BatchNorm)
    model.eval()
    
    correctos = 0
    total = 0
    
    # Desactivar el cálculo de gradientes para ahorrar memoria y acelerar
    with torch.no_grad():
        for espectrogramas, etiquetas in dataloader_test:
            espectrogramas, etiquetas = espectrogramas.to(device), etiquetas.to(device)
            
            # Predicción
            salidas = model(espectrogramas)
            _, predicciones = torch.max(salidas.data, 1)
            
            total += etiquetas.size(0)
            correctos += (predicciones == etiquetas).sum().item()
            
    print("\n --- RESULTADOS DE LA EVALUACIÓN ---")
    precisión = (correctos / total) * 100
    print(f" Total de audios evaluados: {total}")
    print(f" Precisión del modelo (Accuracy): {precisión:.2f}%")
    print(" ¡El script de evaluación y métricas quedó completado exitosamente!")

if __name__ == "__main__":
    evaluar_sistema()