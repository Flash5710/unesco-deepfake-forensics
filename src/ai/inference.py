import os
import torch
from src.ai.model import DeepfakeAudioCNN

class DeepfakeDetectorInference:
    def __init__(self, model_path="src/ai/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepfakeAudioCNN().to(self.device)
        
       
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f" Pesos optimizados cargados desde {model_path}")
        else:
            print(" No se encontró checkpoint previo. Ejecutando con pesos iniciales.")
            
        self.model.eval() # Modo evaluación para máxima velocidad de inferencia

    def predecir_espectrograma(self, espectrograma_tensor):
        """
        Recibe un tensor de espectrograma y retorna la clase (0: Real, 1: Fake) 
        junto con la probabilidad calculada.
        """
        with torch.no_grad():
            espectrograma_tensor = espectrograma_tensor.to(self.device)
            # Asegurar dimensión de lote: (1, 1, 128, frames)
            if espectrograma_tensor.dim() == 3:
                espectrograma_tensor = espectrograma_tensor.unsqueeze(0)
                
            salidas = self.model(espectrograma_tensor)
            probabilidades = torch.softmax(salidas, dim=1)
            clase_predicha = torch.argmax(probabilidades, dim=1).item()
            confianza = probabilidades[0][clase_predicha].item() * 100
            
        etiqueta = "FAKE" if clase_predicha == 1 else "REAL"
        return etiqueta, confianza

if __name__ == "__main__":
    print(" Probando el módulo de inferencia rápida (Martes - Semana 2)...")
    detector = DeepfakeDetectorInference()
    
    # Simulación de un espectrograma entrante
    audio_dummy = torch.randn(1, 128, 44)
    etiqueta, confianza = detector.predecir_espectrograma(audio_dummy)
    print(f" Resultado de Inferencia: [{etiqueta}] con {confianza:.2f}% de confianza.")