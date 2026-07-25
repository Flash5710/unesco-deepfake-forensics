import librosa
import numpy as np
import torch

def cargar_y_normalizar_audio(ruta_audio, target_sr=16000):
    """Normaliza la amplitud del audio acústico (Tarea del Matemático - Semana 1)"""
    y, sr = librosa.load(ruta_audio, sr=target_sr)
    # Escala matemáticamente los valores del audio entre -1 y 1
    y = librosa.util.normalize(y)
    return y, sr

def convertir_a_tensor_pytorch(ruta_audio):
    """Convierte el audio a un tensor matemático (Matriz 2D) para la IA"""
    y, sr = librosa.load(ruta_audio, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Recortar o rellenar la matriz con ceros para que mida exactamente 44 frames
    max_len = 44
    if mel_db.shape[1] > max_len:
        mel_db = mel_db[:, :max_len]
    else:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    # Convertir a tensor de PyTorch y agregar dimensión de canal: Forma (1, 128, 44)
    tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
    return tensor