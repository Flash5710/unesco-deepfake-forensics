import os
import torch
from src.ai.model import DeepfakeAudioCNN

def inicializar_modelo_ui(model_path="src/ai/best_model.pth"):
    """
    Funcion de inicializacion para la UI (Streamlit/Web).
    Carga y entrega los artefactos del modelo listos en memoria.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = DeepfakeAudioCNN().to(device)
    
    
    if os.path.exists(model_path):
        try:
            modelo.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Artefactos del modelo cargados exitosamente desde {model_path}")
        except Exception as e:
            print(f" Error cargando artefactos: {e}. Usando modelo base inicializado.")
    else:
        print("ℹ️ Artefactos preentrenados no encontrados. Modelo cargado con pesos base para la UI.")
        
    modelo.eval()
    return modelo, device

if __name__ == "__main__":
    print(" Probando inicializacion del modelo para la UI (Lunes - Semana 3)...")
    modelo_cargado, dev = inicializar_modelo_ui()
    print(f" Modelo listo para la UI en dispositivo: {dev}")