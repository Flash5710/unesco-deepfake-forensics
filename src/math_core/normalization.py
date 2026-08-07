import librosa
import numpy as np

TARGET_SR = 48000
MAX_FRAMES = 300
N_MELS = 128


def cargar_y_normalizar_audio(ruta_audio, target_sr=TARGET_SR):
    """Normaliza la amplitud del audio acústico"""
    y, sr = librosa.load(ruta_audio, sr=target_sr)
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp
    return y, sr


def normalizar_espectrograma(matriz):
    """Z-score normalization por muestra: media=0, std=1."""
    media = np.mean(matriz)
    std = np.std(matriz)
    if std > 1e-8:
        return (matriz - media) / std
    return matriz - media


def convertir_a_tensor_pytorch(matriz, agregar_canal=True, normalizar=True):
    """Convierte matriz espectral a tensor PyTorch con padding/recorte."""
    import torch
    frames_actuales = matriz.shape[1]
    if frames_actuales < MAX_FRAMES:
        matriz = np.pad(
            matriz,
            ((0, 0), (0, MAX_FRAMES - frames_actuales)),
            mode='constant',
            constant_values=-80.0
        )
    elif frames_actuales > MAX_FRAMES:
        matriz = matriz[:, :MAX_FRAMES]
    if normalizar:
        matriz = normalizar_espectrograma(matriz)
    tensor = torch.from_numpy(matriz).float()
    if agregar_canal:
        tensor = tensor.unsqueeze(0)
    return tensor
