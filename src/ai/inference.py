# inference.py
import os
import torch
from src.ai.model import DeepfakeAudioCNN

class DeepfakeDetectorInference:
    def __init__(self, model_path="src/ai/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepfakeAudioCNN().to(self.device)
        
        if not os.path.exists(model_path):
            # Falla explícita: nunca inferir con pesos aleatorios en un sistema forense
            raise FileNotFoundError(
                f"No se encontró el checkpoint '{model_path}'. "
                f"Abortando: inferir con pesos sin entrenar generaría un dictamen falso."
            )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f" Pesos optimizados cargados desde {model_path}")
        self.model.eval() # Modo evaluación para máxima velocidad de inferencia

    def predecir_espectrograma(self, espectrograma_tensor):
        """
        Recibe un tensor de espectrograma y retorna la clase (0: Real, 1: Fake) 
        junto con la probabilidad calculada.
        """
        with torch.no_grad():
            espectrograma_tensor = espectrograma_tensor.to(self.device)
            # Asegurar dimensión de lote: (1, 1, 128, 300)
            if espectrograma_tensor.dim() == 3:
                espectrograma_tensor = espectrograma_tensor.unsqueeze(0)
                
            salidas = self.model(espectrograma_tensor)
            probabilidades = torch.softmax(salidas, dim=1)
            clase_predicha = torch.argmax(probabilidades, dim=1).item()
            confianza = probabilidades[0][clase_predicha].item() * 100
            
        etiqueta = "FAKE" if clase_predicha == 1 else "REAL"
        return etiqueta, confianza

if __name__ == "__main__":
    print(" Probando el módulo de inferencia rápida...")
    detector = DeepfakeDetectorInference()
    
    # Simulación con la nueva forma: 300 frames
    audio_dummy = torch.randn(1, 128, 300)
    etiqueta, confianza = detector.predecir_espectrograma(audio_dummy)
    print(f" Resultado de Inferencia: [{etiqueta}] con {confianza:.2f}% de confianza.")
