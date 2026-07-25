import torch
from src.ai.inference import DeepfakeDetectorInference

class AIPipeline:
    """
    Pipeline principal de IA que conecta la entrada de datos 
    del módulo de matemáticas/física con el motor de inferencia.
    """
    def __init__(self):
        print(" Inicializando Pipeline de Integración de IA (Miércoles - Semana 2)...")
        self.detector = DeepfakeDetectorInference()

    def procesar_audio_tensor(self, tensor_espectrograma):
        """
        Recibe el tensor generado por los módulos previos y retorna 
        un diccionario estructurado con los resultados del análisis.
        """
        if tensor_espectrograma is None:
            return {
                "estado": "ERROR",
                "mensaje": "El tensor de entrada está vacío o es inválido."
            }

        etiqueta, confianza = self.detector.predecir_espectrograma(tensor_espectrograma)
    
        # Estructura de respuesta lista para consumir por Backend/Web
        resultado = {
            "estado": "EXITO",
            "diagnostico": etiqueta, # "REAL" o "FAKE"
            "confianza_porcentaje": round(confianza, 2),
            "es_deepfake": True if etiqueta == "FAKE" else False
        }
        
        return resultado

if __name__ == "__main__":
    
    pipeline = AIPipeline()
    
    # Simulación de un espectrograma de Mel procesado 
    tensor_prueba = torch.randn(1, 128, 44)
    respuesta = pipeline.procesar_audio_tensor(tensor_prueba)
    
    print("\n Respuesta estructurada del Pipeline:")
    print(respuesta)
    print(" ¡Pipeline de IA integrado y probado correctamente!")