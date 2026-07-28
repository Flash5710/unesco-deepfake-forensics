import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ai.custom_dataset import DeepfakeAudioDataset, TARGET_SR
from src.ai.model import DeepfakeAudioCNN

BATCH_SIZE = 16
N_MELS = 128
MAX_FRAMES = 300

def precomputar_audios_crudos(dataset):
    from src.physics.audio_processing import cargar_y_normalizar_audio
    from src.ai.custom_dataset import TARGET_SR, silenciar

    audios = []
    labels = []
    for i in range(len(dataset)):
        file_path = dataset.files[i]
        lbl = dataset.labels[i]
        with silenciar():
            y, _ = cargar_y_normalizar_audio(file_path, target_sr=TARGET_SR)
        audios.append(y.astype(np.float32))
        labels.append(lbl)
    return audios, labels

class PrecomputedRawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, raw_audios, labels, augment=False):
        self.raw_audios = raw_audios
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = self.raw_audios[idx].copy()
        lbl = self.labels[idx]
        audio_tensor = torch.from_numpy(y).float()
        return audio_tensor, torch.tensor(lbl, dtype=torch.long)

def collate_audios(batch):
    audios, labels = zip(*batch)
    max_len = max(a.shape[0] for a in audios)
    padded = []
    for a in audios:
        if a.shape[0] < max_len:
            a = F.pad(a, (0, max_len - a.shape[0]))
        padded.append(a)
    return torch.stack(padded), torch.stack(labels)

def evaluar_modelo(modelo_path, nombre):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Evaluando: {nombre}")
    print(f"{'='*55}")
    print(f"  Dispositivo: {device}")

    ds_raw = DeepfakeAudioDataset(base_dir="./data", split="test", augment=False, normalizar=True)
    if len(ds_raw) == 0:
        print("  No hay datos de test.")
        return None

    print(f"  Precomputando audios crudos ({len(ds_raw)} muestras)...")
    audios, labels = precomputar_audios_crudos(ds_raw)
    ds = PrecomputedRawAudioDataset(audios, labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_audios)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=2048, hop_length=512, n_mels=N_MELS
    ).to(device)
    db_transform = torchaudio.transforms.AmplitudeToDB().to(device)

    model = DeepfakeAudioCNN().to(device)
    if not os.path.exists(modelo_path):
        print(f"  ERROR: No se encontro '{modelo_path}'")
        return None
    model.load_state_dict(torch.load(modelo_path, map_location=device))
    model.eval()

    todas_preds = []
    todas_reales = []

    with torch.no_grad():
        for audios_batch, etiquetas in loader:
            etiquetas = etiquetas.to(device)
            audios_batch = audios_batch.to(device)
            mel = mel_transform(audios_batch)
            mel_db = db_transform(mel)

            B, n_mels, T = mel_db.shape
            if T < MAX_FRAMES:
                pad = MAX_FRAMES - T
                mel_db = F.pad(mel_db, (0, pad), mode='constant', value=mel_db.min().item())
            elif T > MAX_FRAMES:
                mel_db = mel_db[:, :, :MAX_FRAMES]

            mean = mel_db.mean(dim=(1, 2), keepdim=True)
            std = mel_db.std(dim=(1, 2), keepdim=True)
            std = std.clamp(min=1e-8)
            mel_db = (mel_db - mean) / std
            mel_db = mel_db.unsqueeze(1)

            salidas = model(mel_db)
            _, preds = torch.max(salidas, 1)
            todas_preds.extend(preds.cpu().tolist())
            todas_reales.extend(etiquetas.tolist())

    todas_reales = np.array(todas_reales)
    todas_preds = np.array(todas_preds)

    VP = int(np.sum((todas_preds == 1) & (todas_reales == 1)))
    FP = int(np.sum((todas_preds == 1) & (todas_reales == 0)))
    VN = int(np.sum((todas_preds == 0) & (todas_reales == 0)))
    FN = int(np.sum((todas_preds == 0) & (todas_reales == 1)))

    total = len(todas_reales)
    accuracy = 100.0 * (VP + VN) / total if total > 0 else 0.0
    precision_fake = 100.0 * VP / (VP + FP) if (VP + FP) > 0 else 0.0
    recall_fake = 100.0 * VP / (VP + FN) if (VP + FN) > 0 else 0.0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0.0

    print(f"  Accuracy:      {accuracy:.2f}%")
    print(f"  Precision Fake:{precision_fake:.2f}%")
    print(f"  Recall Fake:   {recall_fake:.2f}%")
    print(f"  F1-Score Fake: {f1_fake:.2f}%")
    print(f"  VP={VP}  FP={FP}  VN={VN}  FN={FN}")

    return {
        "nombre": nombre,
        "accuracy": accuracy,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "VP": VP, "FP": FP, "VN": VN, "FN": FN,
        "total": total,
    }


if __name__ == "__main__":
    results = []
    for path, name in [
        ("src/ai/best_model_v2_augmented.pth", "v2 Aug (acoustic aug)"),
        ("src/ai/best_model.pth", "v1 Original (sin aug)"),
    ]:
        r = evaluar_modelo(path, name)
        if r:
            results.append(r)

    if len(results) == 2:
        a, b = results
        print(f"\n{'='*65}")
        print(f"  COMPARACION FINAL - Test Set ({a['total']} muestras)")
        print(f"{'='*65}")
        print(f"  {'Metrica':<25} {a['nombre']:<20} {b['nombre']:<20}")
        print(f"  {'-'*25} {'-'*20} {'-'*20}")
        print(f"  {'Accuracy':<25} {a['accuracy']:<20.2f} {b['accuracy']:<20.2f}")
        print(f"  {'Precision Fake':<25} {a['precision_fake']:<20.2f} {b['precision_fake']:<20.2f}")
        print(f"  {'Recall Fake':<25} {a['recall_fake']:<20.2f} {b['recall_fake']:<20.2f}")
        print(f"  {'F1-Score Fake':<25} {a['f1_fake']:<20.2f} {b['f1_fake']:<20.2f}")
        print(f"  {'-'*25} {'-'*20} {'-'*20}")
        print(f"  {'VP (True Positives)':<25} {a['VP']:<20} {b['VP']:<20}")
        print(f"  {'VN (True Negatives)':<25} {a['VN']:<20} {b['VN']:<20}")
        print(f"  {'FP (False Positives)':<25} {a['FP']:<20} {b['FP']:<20}")
        print(f"  {'FN (False Negatives)':<25} {a['FN']:<20} {b['FN']:<20}")
        print(f"{'='*65}")