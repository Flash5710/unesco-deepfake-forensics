# predict.py
import os
import numpy as np
import torch
from src.ai.model import DeepfakeAudioCNN
from src.physics.audio_processing import (
    cargar_y_normalizar_audio,
    calcular_stft_y_mel,
    convertir_a_tensor_pytorch,
)


def normalizar_espectrograma(mel_db):
    media = np.mean(mel_db)
    std = np.std(mel_db)
    if std > 1e-8:
        return (mel_db - media) / std
    return mel_db - media

FRAMES_FIJOS = 300


def predecir_etiqueta(audio_input, model_path="src/ai/best_model.pth"):
    """
    Convierte una entrada de audio en una etiqueta ("REAL" o "FAKE").

    `audio_input` puede ser:
      - Una ruta (str) a un archivo .wav/.mp3, tal como llega desde la UI.
      - Un tensor ya procesado (torch.Tensor), útil para tests/pruebas de estrés.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelo = DeepfakeAudioCNN().to(device)
    
    if model_path:
        try:
            modelo.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f" No se cargaron pesos externos: {e}")
            
    modelo.eval()
    
    # --- Pipeline modular para entradas del mundo real (ruta de audio) ---
    if isinstance(audio_input, str):
        # 1. Extraer señal (y) y frecuencia (sr) forzando 48000 Hz
        y, sr = cargar_y_normalizar_audio(audio_input, target_sr=16000)
        
        mel_db, _, _ = calcular_stft_y_mel(y, sr)
        mel_db = normalizar_espectrograma(mel_db)
        espectrograma_tensor = convertir_a_tensor_pytorch(mel_db, agregar_canal=True)
    else:
        espectrograma_tensor = audio_input
    
    # Asegurar que el tensor tenga la forma adecuada (Batch, Channel, Frecuencia, Tiempo)
    if espectrograma_tensor.dim() == 2:  # (128, frames)
        espectrograma_tensor = espectrograma_tensor.unsqueeze(0).unsqueeze(0)
    elif espectrograma_tensor.dim() == 3:  # (1, 128, frames)
        espectrograma_tensor = espectrograma_tensor.unsqueeze(0)
        
    espectrograma_tensor = espectrograma_tensor.to(device)
    
    with torch.no_grad():
        salida = modelo(espectrograma_tensor)
        probabilidades = torch.softmax(salida, dim=1)
        clase_idx = torch.argmax(probabilidades, dim=1).item()
        confianza = probabilidades[0][clase_idx].item() * 100

    etiqueta = "FAKE" if clase_idx == 1 else "REAL"
    return etiqueta, confianza

if __name__ == "__main__":
    print(" Probando script auxiliar de predicción...")
    tensor_prueba = torch.randn(1, 128, FRAMES_FIJOS)
    etiqueta, confianza = predecir_etiqueta(tensor_prueba)
    print(f" Espectrograma convertido a etiqueta: [{etiqueta}] ({confianza:.2f}% de confianza)")
