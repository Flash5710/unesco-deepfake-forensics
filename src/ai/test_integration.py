import unittest
import torch
from src.ai.model import DeepfakeAudioCNN
from src.ai.inference import DeepfakeDetectorInference
from src.ai.pipeline import AIPipeline

class TestIAIntegration(unittest.TestCase):
    
    def setUp(self):
        """Inicializa los módulos para cada prueba."""
        self.pipeline = AIPipeline()
        self.detector = DeepfakeDetectorInference()
        self.modelo = DeepfakeAudioCNN()

    def test_modelo_forward(self):
        """Valida que la CNN procese un lote sin errores de forma."""
        tensor_entrada = torch.randn(2, 1, 128, 300)
        salida = self.modelo(tensor_entrada)
        self.assertEqual(salida.shape, (2, 2), "La salida del modelo debe tener la forma [batch_size, 2]")

    def test_pipeline_respuesta_correcta(self):
        """Valida que el pipeline devuelva un diagnóstico estructurado válido."""
        tensor_entrada = torch.randn(1, 128, 300)
        resultado = self.pipeline.procesar_audio_tensor(tensor_entrada)
        
        self.assertEqual(resultado["estado"], "EXITO")
        self.assertIn(resultado["diagnostico"], ["REAL", "FAKE"])
        self.assertIsInstance(resultado["confianza_porcentaje"], float)

    def test_pipeline_manejo_errores(self):
        """Valida que el pipeline maneje adecuadamente entradas nulas."""
        resultado = self.pipeline.procesar_audio_tensor(None)
        self.assertEqual(resultado["estado"], "ERROR")

if __name__ == "__main__":
    print(" Ejecutando Pruebas de Integración Extremo a Extremo (Jueves - Semana 2)...")
    unittest.main()
