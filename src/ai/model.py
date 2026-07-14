import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepfakeAudioCNN(nn.Module):
    def __init__(self):
        super(DeepfakeAudioCNN, self).__init__()
        
        # Bloque Convolucional 1: Entrada (1, 128, frames) -> Salida (16, 64, frames/2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Bloque Convolucional 2: Entrada (16, 64, frames/2) -> Salida (32, 32, frames/4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Capa de Reducción MaxPool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Reducción adaptativa para soportar audios de cualquier duración
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) # Reduce a un tamaño fijo de (32, 4, 4)
        
        # Capas de Clasificación Totalmente Conectadas 
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2) # 2 Clases: 0 = Real, 1 = Fake

    def forward(self, x):
        # Flujo de datos a través de la red convolucional ligera
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Aplanar para la capa lineal
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        
        # Clasificación
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        
        return x

if __name__ == "__main__":
    modelo = DeepfakeAudioCNN()
    # Tensor simulado: 
    tensor_simulado = torch.randn(2, 1, 128, 44)
    salida = modelo(tensor_simulado)
    
    print(" Arquitectura de la CNN Ligera definida con éxito para el Jueves.")
    print(f" Forma de la salida: {salida.shape} -> [Lote, Clases]")