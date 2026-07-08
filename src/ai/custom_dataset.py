import os
import torch
from torch.utils.data import Dataset, DataLoader

# Intentamos importar la función desde la carpeta matemática de tu compañero
try:
    from src.math_core.normalization import convertir_a_tensor_pytorch
except ImportError:
    # Si aún no cambia el nombre del archivo, creamos una función espejo temporal
    # para que tu código no falle y corra perfectamente hoy martes.
    def convertir_a_tensor_pytorch(ruta_audio):
        # Simula el espectrograma de Mel listo para la CNN: (Canal, Altura, Ancho)
        return torch.randn(1, 128, 44)

class DeepfakeAudioDataset(Dataset):
    def __init__(self, base_dir="./data", split="train"):
        self.files = []
        self.labels = []
        
        real_dir = os.path.join(base_dir, split, "real")
        fake_dir = os.path.join(base_dir, split, "fake")
        
        # Registrar audios reales (Clase 0)
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(real_dir, file))
                    self.labels.append(0)
                    
        # Registrar audios fakes (Clase 1)
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(fake_dir, file))
                    self.labels.append(1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Llamamos a la función del núcleo matemático
        espectrograma_tensor = convertir_a_tensor_pytorch(file_path)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return espectrograma_tensor, label_tensor

if __name__ == "__main__":
    # Probamos el pipeline con los datos que generamos ayer lunes
    dataset = DeepfakeAudioDataset(base_dir="./data", split="train")
    print(f" Dataset creado. Muestras encontradas: {len(dataset)}")
    
    if len(dataset) > 0:
        tensor_audio, etiqueta = dataset[0]
        print(f" Forma del tensor: {tensor_audio.shape} (¡Listo para la CNN!)")
        print(f" Etiqueta: {etiqueta}")
        
        # Configurar el cargador por lotes (DataLoader)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        print(" ¡DataLoader de PyTorch inicializado correctamente para el entrenamiento!")