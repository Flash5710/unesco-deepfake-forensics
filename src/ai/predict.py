import torch
from src.ai.model import DeepfakeAudioCNN

def predecir_etiqueta(espectrograma_tensor, model_path=None):
    """
    Convierte un espectrograma_tensor de entrada en una etiqueta ("REAL" o "FAKE").
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelo = DeepfakeAudioCNN().to(device)
    
    
    if model_path:
        try:
            modelo.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f" No se cargaron pesos externos: {e}")
            
    modelo.eval()
    
    #  Asegurar que el tensor tenga la forma adecuada (Batch, Channel, Frecuencia, Tiempo)
    if espectrograma_tensor.dim() == 2: # (128, frames)
        espectrograma_tensor = espectrograma_tensor.unsqueeze(0).unsqueeze(0)
    elif espectrograma_tensor.dim() == 3: # (1, 128, frames)
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
    print(" Probando script auxiliar de predicción (Viernes)...")
    tensor_prueba = torch.randn(1, 128, 44)
    etiqueta, confianza = predecir_etiqueta(tensor_prueba)
    print(f" Espectrograma convertido a etiqueta: [{etiqueta}] ({confianza:.2f}% de confianza)")