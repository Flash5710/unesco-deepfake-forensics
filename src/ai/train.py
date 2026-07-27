import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# === Entrenamiento de Produccion ===
N_EPOCHS = 30
BATCH_SIZE = 32
VAL_SPLIT = 0.15
# LR moderado para fine-tuning desde 48kHz -> 16kHz
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE_EARLY = 6
PATIENCE_LR = 3
FACTOR_LR = 0.5
MAX_SAMPLES = 100000

MODEL_SAVE_PATH = "src/ai/best_model.pth"
MODEL_WAV_PATH = "src/ai/best_model_wav.pth"


class PrecomputedDataset(Dataset):
    def __init__(self, spectrograms, labels, augment=False):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx].clone()
        if self.augment:
            from src.ai.custom_dataset import augmentar_espectrograma_tensor
            spec = augmentar_espectrograma_tensor(spec)
        return spec, torch.tensor(self.labels[idx], dtype=torch.long)


def precomputar_espectrogramas(dataset):
    specs = []
    labels = []
    t0 = time.time()
    n = len(dataset)
    for i in range(n):
        spec, lbl = dataset[i]
        specs.append(spec.cpu())
        labels.append(lbl)
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            tasa = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / tasa if tasa > 0 else 0
            print(f"  Precomputo: {i+1}/{n} ({elapsed:.0f}s, {tasa:.1f}/s, ETA {eta:.0f}s)")
            sys.stdout.flush()
    print(f"Precomputo completo: {len(specs)} muestras en {time.time()-t0:.0f}s")
    return specs, labels


def cargar_pesos_pretreinados(model, ruta):
    if not os.path.exists(ruta):
        print(f"[Fase 4] No se encontro {ruta}. Entrenando desde cero.")
        return model, False
    try:
        state_dict = torch.load(ruta, map_location='cpu', weights_only=True)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"[Fase 4] Capas nuevas (inicializadas aleatoriamente): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[Fase 4] Capas ignoradas del checkpoint: {incompatible.unexpected_keys}")
        print(f"[Fase 4] Pesos cargados desde {ruta}. Fine-tuning con LR={LEARNING_RATE}")
        return model, True
    except Exception as e:
        print(f"[Fase 4] Error cargando pesos: {e}. Entrenando desde cero.")
        return model, False


def evaluar_en_test(model, device):
    from src.ai.custom_dataset import DeepfakeAudioDataset
    ds_test = DeepfakeAudioDataset(base_dir="./data", split="test", augment=False, normalizar=True)
    if len(ds_test) == 0:
        print("[Fase 5] No hay datos de test para evaluar.")
        return

    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    todas_preds = []
    todas_reales = []

    with torch.no_grad():
        for espectrogramas, etiquetas in loader_test:
            espectrogramas = espectrogramas.to(device)
            salidas = model(espectrogramas)
            _, preds = torch.max(salidas, 1)
            todas_preds.extend(preds.cpu().tolist())
            todas_reales.extend(etiquetas.tolist())

    todas_reales = np.array(todas_reales)
    todas_preds = np.array(todas_preds)

    VP = np.sum((todas_preds == 1) & (todas_reales == 1))
    FP = np.sum((todas_preds == 1) & (todas_reales == 0))
    VN = np.sum((todas_preds == 0) & (todas_reales == 0))
    FN = np.sum((todas_preds == 0) & (todas_reales == 1))

    total = len(todas_reales)
    accuracy = 100.0 * (VP + VN) / total if total > 0 else 0.0
    precision_fake = 100.0 * VP / (VP + FP) if (VP + FP) > 0 else 0.0
    recall_fake = 100.0 * VP / (VP + FN) if (VP + FN) > 0 else 0.0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  Fase 5: Validacion en Test ({len(ds_test)} muestras de .mp4)")
    print(f"{'='*55}")
    print(f"  Matriz de Confusion:")
    print(f"                    Predicho")
    print(f"                  Real    Fake")
    print(f"  Real     {VN:>6d}  {FP:>6d}")
    print(f"  Fake     {FN:>6d}  {VP:>6d}")
    print(f"{'-'*55}")
    print(f"  Accuracy:      {accuracy:.2f}%")
    print(f"  Precision Fake:{precision_fake:.2f}%")
    print(f"  Recall Fake:   {recall_fake:.2f}%  <-- CRITICO")
    print(f"  F1-Score Fake: {f1_fake:.2f}%")
    print(f"{'='*55}\n")


def ejecutar_entrenamiento():
    from src.ai.custom_dataset import DeepfakeAudioDataset
    from src.ai.model import DeepfakeAudioCNN

    dataset_full = DeepfakeAudioDataset(base_dir="./data", split="train", augment=False, normalizar=True)
    if len(dataset_full) == 0:
        print("No se encontraron audios en ./data/train. Abortando.")
        return

    print(f"Dataset completo: {len(dataset_full)} muestras")

    n_val = max(1, int(len(dataset_full) * VAL_SPLIT))
    indices = np.random.RandomState(42).permutation(len(dataset_full))

    if len(dataset_full) > MAX_SAMPLES:
        indices = indices[:MAX_SAMPLES]
        n_val = max(1, int(MAX_SAMPLES * VAL_SPLIT))
        print(f"[Fase 3] Fast Dev Run: {MAX_SAMPLES} muestras, {N_EPOCHS} epocas")

    idx_val = indices[:n_val]
    idx_train = indices[n_val:]

    files_train = [dataset_full.files[i] for i in idx_train]
    labels_train = [dataset_full.labels[i] for i in idx_train]
    files_val = [dataset_full.files[i] for i in idx_val]
    labels_val = [dataset_full.labels[i] for i in idx_val]

    ds_train_raw = DeepfakeAudioDataset(file_list=files_train, label_list=labels_train, augment=False, normalizar=True)
    ds_val_raw = DeepfakeAudioDataset(file_list=files_val, label_list=labels_val, augment=False, normalizar=True)

    print(f"Precomputando espectrogramas...")
    print(f"Train: {len(ds_train_raw)} muestras")
    specs_train, labels_train = precomputar_espectrogramas(ds_train_raw)
    print(f"Val: {len(ds_val_raw)} muestras")
    specs_val, labels_val = precomputar_espectrogramas(ds_val_raw)

    ds_train = PrecomputedDataset(specs_train, labels_train, augment=True)
    ds_val = PrecomputedDataset(specs_val, labels_val, augment=False)
    print(f"\nTrain: {len(ds_train)} | Val: {len(ds_val)}")

    clase_counts = np.bincount(labels_train)
    print(f"Distribucion: Real={clase_counts[0]}, Fake={clase_counts[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cpu":
        print("  [WARN] PyTorch CPU-only. Para usar GTX 1650: instalar Python 3.12-3.13 y pip install torch --index-url https://download.pytorch.org/whl/cu124")

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepfakeAudioCNN().to(device)
    model, pretrained = cargar_pesos_pretreinados(model, MODEL_WAV_PATH)

    lr = LEARNING_RATE
    if not pretrained:
        lr = 0.001
        print(f"[Fase 4] Sin pesos pre-entrenados. Entrenando desde cero con LR={lr}")

    peso_clases = torch.tensor([1.0 / clase_counts[0], 1.0 / clase_counts[1]], dtype=torch.float).to(device)
    peso_clases = peso_clases / peso_clases.sum()
    criterion = nn.CrossEntropyLoss(weight=peso_clases)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR_LR, patience=PATIENCE_LR)

    mejor_loss_val = float('inf')
    epochs_sin_mejora = 0
    t_inicio = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        t_epoca = time.time()

        for espectrogramas, etiquetas in loader_train:
            espectrogramas, etiquetas = espectrogramas.to(device), etiquetas.to(device)
            optimizer.zero_grad()
            predicciones = model(espectrogramas)
            loss = criterion(predicciones, etiquetas)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        loss_train_prom = running_loss / len(loader_train)

        model.eval()
        loss_val_acum = 0.0
        correctos = 0
        total = 0

        with torch.no_grad():
            for espectrogramas, etiquetas in loader_val:
                espectrogramas, etiquetas = espectrogramas.to(device), etiquetas.to(device)
                salidas = model(espectrogramas)
                loss = criterion(salidas, etiquetas)
                loss_val_acum += loss.item()
                _, preds = torch.max(salidas, 1)
                total += etiquetas.size(0)
                correctos += (preds == etiquetas).sum().item()

        loss_val_prom = loss_val_acum / len(loader_val)
        precision_val = 100.0 * correctos / total if total > 0 else 0.0

        lr_actual = optimizer.param_groups[0]['lr']
        t_epoca = time.time() - t_epoca
        print(f"Epoch [{epoch:2d}/{N_EPOCHS}]  "
              f"Loss T: {loss_train_prom:.4f}  "
              f"Loss V: {loss_val_prom:.4f}  "
              f"Acc V: {precision_val:.2f}%  "
              f"LR: {lr_actual:.6f}  "
              f"{t_epoca:.0f}s")
        sys.stdout.flush()

        scheduler.step(loss_val_prom)

        if loss_val_prom < mejor_loss_val:
            mejor_loss_val = loss_val_prom
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_sin_mejora = 0
            print(f"  -> Mejor modelo (Val Loss: {mejor_loss_val:.4f})")
        else:
            epochs_sin_mejora += 1

        if epochs_sin_mejora >= PATIENCE_EARLY:
            print(f"  -> Early stopping en epoca {epoch}")
            break

    total_t = time.time() - t_inicio
    print(f"\nEntrenamiento completado en {total_t:.0f}s. Mejor Val Loss: {mejor_loss_val:.4f}")
    print(f"Pesos en: {MODEL_SAVE_PATH}")

    evaluar_en_test(model, device)


if __name__ == "__main__":
    ejecutar_entrenamiento()
