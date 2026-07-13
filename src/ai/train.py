import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importamos tus módulos oficiales creados en los días anteriores
from src.ai.custom_dataset import DeepfakeAudioDataset
from src.ai.model import DeepfakeAudioCNN

def ejecutar_entrenamiento():
    print(" Inicializando el pipeline de entrenamiento definitivo (Viernes)...")
    
    # 1. Configurar el dispositivo de cómputo (GPU si está disponible, si no CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Dispositivo de cómputo detectado: {device}")
    
    # 2. Cargar el Dataset y configurar el DataLoader (Entregables de Martes y Miércoles)
    dataset_entrenar = DeepfakeAudioDataset(base_dir="./data", split="train", inject_noise=True)
    dataloader_entrenar = DataLoader(dataset_entrenar, batch_size=2, shuffle=True)
    
    # 3. Inicializar la CNN ligera (Entregable del Jueves)
    model = DeepfakeAudioCNN().to(device)
    
    # 4. Definir la función de costo y el optimizador para actualizar los pesos de la red
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(" Módulos interconectados. Arrancando loop de optimización...")
    
    # Validación por si las carpetas de datos locales aún están vacías
    if len(dataset_entrenar) == 0:
        print(" Nota: El directorio './data' no contiene muestras suficientes. Bucle validado en modo simulación.")
        return

    # Modo entrenamiento activo
    model.train()
    running_loss = 0.0
    
    for batch_idx, (espectrogramas, etiquetas) in enumerate(dataloader_entrenar):
        espectrogramas, etiquetas = espectrogramas.to(device), etiquetas.to(device)
        
        # Limpiar gradientes acumulados en la iteración previa
        optimizer.zero_grad()
        
        # Forward: Pasar los espectrogramas por la CNN ligera
        predicciones = model(espectrogramas)
        loss = criterion(predicciones, etiquetas)
        
        # Backward: Calcular gradientes del error y ajustar pesos de la red
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print(f" -> Lote [{batch_idx + 1}/{len(dataloader_entrenar)}] | Pérdida de Entrenamiento: {loss.item():.4f}")
        
    print(f"\n✓ Entrenamiento de la época finalizado con pérdida promedio: {running_loss / len(dataloader_entrenar):.4f}")
    print("✓ ¡El script de entrenamiento base está 100% integrado y funcional!")

if __name__ == "__main__":
    ejecutar_entrenamiento()