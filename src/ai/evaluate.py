import os
import torch
from torch.utils.data import DataLoader

from src.ai.custom_dataset import DeepfakeAudioDataset
from src.ai.model import DeepfakeAudioCNN

MODEL_PATH = "src/ai/best_model.pth"


def evaluar_sistema():
    print("Inicializando evaluacion...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en: {device}")

    dataset_test = DeepfakeAudioDataset(base_dir="./data", split="test", augment=False, normalizar=True)
    dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

    model = DeepfakeAudioCNN().to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontro '{MODEL_PATH}'. "
            f"Evaluar sin pesos daria precision falsa cercana al 50%."
        )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Pesos cargados desde {MODEL_PATH}")

    if len(dataset_test) == 0:
        print("No hay datos de test.")
        return

    model.eval()

    correctos = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for espectrogramas, etiquetas in dataloader_test:
            espectrogramas, etiquetas = espectrogramas.to(device), etiquetas.to(device)

            salidas = model(espectrogramas)
            _, predicciones = torch.max(salidas.data, 1)

            total += etiquetas.size(0)
            correctos += (predicciones == etiquetas).sum().item()

            tp += ((predicciones == 1) & (etiquetas == 1)).sum().item()
            tn += ((predicciones == 0) & (etiquetas == 0)).sum().item()
            fp += ((predicciones == 1) & (etiquetas == 0)).sum().item()
            fn += ((predicciones == 0) & (etiquetas == 1)).sum().item()

    precision = (correctos / total) * 100 if total > 0 else 0
    sensibilidad = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    especificidad = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0

    print("\n--- RESULTADOS DE EVALUACION ---")
    print(f"Total evaluados: {total}")
    print(f"Precision (Accuracy): {precision:.2f}%")
    print(f"Sensibilidad (TPR):  {sensibilidad:.2f}%")
    print(f"Especificidad (TNR): {especificidad:.2f}%")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")


if __name__ == "__main__":
    evaluar_sistema()
