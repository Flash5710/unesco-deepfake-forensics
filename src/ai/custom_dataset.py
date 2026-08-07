import os
import sys
import io
import contextlib
import torch
import numpy as np
from torch.utils.data import Dataset


import warnings
warnings.filterwarnings('ignore')

@contextlib.contextmanager
def silenciar():
    old_out, old_err = sys.stdout, sys.stderr
    old_filters = warnings.filters[:]
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    warnings.filterwarnings('ignore')
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        warnings.filters = old_filters


from src.physics.audio_processing import (
    cargar_y_normalizar_audio,
    calcular_stft_y_mel,
    convertir_a_tensor_pytorch,
)

N_MELS = 128
MAX_FRAMES = 300
VALOR_SILENCIO_DB = -80.0
TARGET_SR = 16000
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache")
CONFIG_HASH = f"{TARGET_SR}_{N_MELS}_{MAX_FRAMES}"
CACHE_VERSION_FILE = os.path.join(CACHE_DIR, "cache_version.txt")


def invalidar_cache_si_cambio_config():
    if not os.path.exists(CACHE_DIR):
        return
    version_path = CACHE_VERSION_FILE
    if os.path.exists(version_path):
        with open(version_path) as f:
            saved = f.read().strip()
        if saved == CONFIG_HASH:
            return
    import shutil
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(version_path, "w") as f:
        f.write(CONFIG_HASH)


def normalizar_espectrograma(mel_db):
    media = np.mean(mel_db)
    std = np.std(mel_db)
    if std > 1e-8:
        return (mel_db - media) / std
    return mel_db - media


def augmentar_espectrograma_tensor(tensor, intensidad=1.0):
    if intensidad <= 0 or tensor.numel() == 0:
        return tensor

    if tensor.dim() == 2:
        m = tensor.unsqueeze(0)
        squeeze = True
    else:
        m = tensor
        squeeze = False

    B, F, T = m.shape

    if torch.rand(1).item() < 0.4 * intensidad:
        corte = int(torch.randint(N_MELS // 4, N_MELS // 2 + 1, (1,)).item())
        m[:, corte:, :] = VALOR_SILENCIO_DB

    if torch.rand(1).item() < 0.5 * intensidad:
        f_ini = int(torch.randint(0, N_MELS - 10, (1,)).item())
        f_an = int(torch.randint(5, 21, (1,)).item())
        m[:, f_ini:min(f_ini + f_an, N_MELS), :] = VALOR_SILENCIO_DB

    if torch.rand(1).item() < 0.3 * intensidad:
        t_ini = int(torch.randint(0, T - 10, (1,)).item())
        t_an = int(torch.randint(5, 31, (1,)).item())
        m[:, :, t_ini:min(t_ini + t_an, T)] = VALOR_SILENCIO_DB

    if torch.rand(1).item() < 0.5 * intensidad:
        ruido = torch.randn_like(m) * (0.1 + 0.4 * torch.rand(1).item())
        m = m + ruido

    if squeeze:
        m = m.squeeze(0)
    return m


class DeepfakeAudioDataset(Dataset):
    def __init__(self, base_dir="./data", split="train", augment=True,
                 file_list=None, label_list=None, normalizar=True):
        invalidar_cache_si_cambio_config()
        self.augment = augment
        self.normalizar = normalizar

        if file_list is not None and label_list is not None:
            self.files = file_list
            self.labels = label_list
        else:
            self.files = []
            self.labels = []
            real_dir = os.path.join(base_dir, split, "real")
            fake_dir = os.path.join(base_dir, split, "fake")

            if os.path.exists(real_dir):
                for file in os.listdir(real_dir):
                    if file.endswith(".wav"):
                        self.files.append(os.path.join(real_dir, file))
                        self.labels.append(0)

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

        cache_key = f"{os.path.basename(file_path)}_{self.normalizar}_{self.augment}"
        cache_path = os.path.join(CACHE_DIR, cache_key.replace(".wav", ".pt").replace(" ", "_"))

        if not self.augment and os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=True)
            return cached, torch.tensor(label, dtype=torch.long)

        os.makedirs(CACHE_DIR, exist_ok=True)

        with silenciar():
            y, sr = cargar_y_normalizar_audio(file_path, target_sr=TARGET_SR)

            if self.augment:
                from src.ai.audio_augmentation import augmentar_audio_crudo
                y = augmentar_audio_crudo(y, sr)

            mel_db, _, _ = calcular_stft_y_mel(y, sr)

            if self.normalizar:
                mel_db = normalizar_espectrograma(mel_db)

            espectrograma_tensor = convertir_a_tensor_pytorch(mel_db, normalizar=False)

            if self.augment:
                espectrograma_tensor = augmentar_espectrograma_tensor(espectrograma_tensor)

        if not self.augment:
            torch.save(espectrograma_tensor, cache_path)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return espectrograma_tensor, label_tensor


if __name__ == "__main__":
    dataset = DeepfakeAudioDataset(base_dir="./data", split="train", augment=True)
    print(f"Dataset creado. Muestras encontradas: {len(dataset)}")

    if len(dataset) > 0:
        tensor_audio, etiqueta = dataset[0]
        print(f"Forma del tensor: {tensor_audio.shape}")
        print(f"Etiqueta: {etiqueta}")
        print(f"Media: {tensor_audio.mean().item():.4f}, Std: {tensor_audio.std().item():.4f}")

        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        print("DataLoader listo.")
