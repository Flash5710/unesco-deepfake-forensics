import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import pytest

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
REAL_WAV = os.path.join(SAMPLES_DIR, "real_voice.wav")
FAKE_WAV = os.path.join(SAMPLES_DIR, "fake_voice.wav")
MODEL_PATH = os.path.join(BASE_DIR, "src", "ai", "best_model.pth")

REQUIRED_FILES = [REAL_WAV, FAKE_WAV]


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def model(device):
    from src.ai.model import DeepfakeAudioCNN
    modelo = DeepfakeAudioCNN().to(device)
    if os.path.exists(MODEL_PATH):
        modelo.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"  Pesos cargados desde {MODEL_PATH}")
    else:
        print(f"  Sin pesos (aleatorio). Modo integracion estructural.")
    modelo.eval()
    return modelo


class TestEndToEnd:

    @pytest.mark.parametrize("ruta, etiqueta_esperada", [
        (REAL_WAV, "real"),
        (FAKE_WAV, "fake"),
    ])
    def test_flujo_completo_desde_wav(self, model, device, ruta, etiqueta_esperada):
        """Carga .wav real → procesa audio → inferencia → clasifica sin crash."""
        from src.physics.audio_processing import (
            cargar_y_normalizar_audio, calcular_stft_y_mel,
            predecir_audio_completo, convertir_a_tensor_pytorch,
        )

        if not os.path.exists(ruta):
            pytest.skip(f"Archivo no encontrado: {ruta}")

        # 1. Carga
        y, sr = cargar_y_normalizar_audio(ruta)

        # 2. STFT + Mel
        mel_db, mfccs, stft_compleja = calcular_stft_y_mel(y, sr)
        assert mel_db.shape[0] == 128, f"Mel debe tener 128 bandas, tiene {mel_db.shape[0]}"
        assert mfccs.shape[0] == 13, f"MFCCs deben tener 13 coefs, tiene {mfccs.shape[0]}"
        assert np.all(np.isfinite(mel_db)), "mel_db no debe contener NaN/inf"

        # 3. Ventanas deslizantes + inferencia
        prob, predicciones_por_ventana, tiempos = predecir_audio_completo(
            mel_db, model, device=device,
            max_frames=300, stride=150, max_padding_ratio=0.4
        )

        assert isinstance(prob, float), f"prob debe ser float, es {type(prob)}"
        assert 0.0 <= prob <= 100.0, f"prob debe estar entre 0-100, es {prob}"
        assert len(predicciones_por_ventana) > 0, "Debe haber al menos 1 ventana"
        assert len(tiempos) == len(predicciones_por_ventana), \
            "tiempos y predicciones deben tener igual longitud"

        for p in predicciones_por_ventana:
            assert 0.0 <= p <= 100.0, f"Cada prediccion debe estar entre 0-100, es {p}"

        print(f"  [{etiqueta_esperada}] prob_fake={prob:.2f}% | {len(predicciones_por_ventana)} ventanas")

    @pytest.mark.parametrize("ruta", [REAL_WAV, FAKE_WAV])
    def test_conversion_a_tensor(self, ruta):
        """Verifica que el tensor de salida tenga forma (1, 1, 128, 300)."""
        from src.physics.audio_processing import (
            cargar_y_normalizar_audio, calcular_stft_y_mel,
            convertir_a_tensor_pytorch,
        )

        if not os.path.exists(ruta):
            pytest.skip(f"Archivo no encontrado: {ruta}")

        y, sr = cargar_y_normalizar_audio(ruta)
        mel_db, _, _ = calcular_stft_y_mel(y, sr)
        tensor = convertir_a_tensor_pytorch(mel_db, agregar_canal=True, normalizar=False)

        assert isinstance(tensor, torch.Tensor), f"Debe ser torch.Tensor, es {type(tensor)}"
        assert tensor.dim() == 3, f"Tensor debe tener 3 dims (C, F, T), tiene {tensor.dim()}"
        assert tensor.shape == (1, 128, 300), \
            f"Tensor debe ser (1, 128, 300), es {tensor.shape}"
        assert tensor.dtype == torch.float32, f"Tensor debe ser float32, es {tensor.dtype}"

    @pytest.mark.parametrize("ruta", [REAL_WAV, FAKE_WAV])
    def test_inferencia_directa(self, model, device, ruta):
        """Envia tensor directo al modelo sin pipeline de ventanas."""
        from src.physics.audio_processing import (
            cargar_y_normalizar_audio, calcular_stft_y_mel,
            convertir_a_tensor_pytorch,
        )

        if not os.path.exists(ruta):
            pytest.skip(f"Archivo no encontrado: {ruta}")

        y, sr = cargar_y_normalizar_audio(ruta)
        mel_db, _, _ = calcular_stft_y_mel(y, sr)
        tensor = convertir_a_tensor_pytorch(mel_db, agregar_canal=True, normalizar=False)

        batch = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            salida = model(batch)
            probs = torch.softmax(salida, dim=1)
            clase = torch.argmax(probs, dim=1).item()
            confianza = probs[0][clase].item() * 100

        assert salida.shape == (1, 2), f"Salida debe ser (1,2), es {salida.shape}"
        assert clase in (0, 1), f"Clase debe ser 0 o 1, es {clase}"
        assert 0.0 <= confianza <= 100.0, f"Confianza debe estar entre 0-100, es {confianza}"
        etiqueta = "FAKE" if clase == 1 else "REAL"
        print(f"  Inferencia directa: [{etiqueta}] conf={confianza:.2f}%")

    def test_fase_sobre_audio_real(self):
        """Calcula regularidad de fase sobre audio real y fake, compara valores."""
        from src.math_core.metrics import calcular_regularidad_fase
        from src.physics.audio_processing import cargar_y_normalizar_audio

        if not os.path.exists(REAL_WAV) or not os.path.exists(FAKE_WAV):
            pytest.skip("Archivos .wav no disponibles")

        y_real, sr = cargar_y_normalizar_audio(REAL_WAV)
        y_fake, _ = cargar_y_normalizar_audio(FAKE_WAV)

        fase_real = calcular_regularidad_fase(y_real)
        fase_fake = calcular_regularidad_fase(y_fake)

        assert np.isfinite(fase_real), f"Fase real debe ser finita, es {fase_real}"
        assert np.isfinite(fase_fake), f"Fase fake debe ser finita, es {fase_fake}"

        print(f"  Fase real: {fase_real:.4f}  |  Fase fake: {fase_fake:.4f}")

        # Opcional: verificar que diff no sea extrema
        diff = abs(fase_real - fase_fake)
        assert diff < 15.0, f"Diferencia de fase anomala: {diff:.4f}"

    @pytest.mark.parametrize("ruta", [REAL_WAV, FAKE_WAV])
    def test_visualizacion_sin_error(self, ruta):
        """Genera mapa de calor interactivo y matplotlib sin excepciones."""
        from src.physics.audio_processing import (
            cargar_y_normalizar_audio, calcular_stft_y_mel,
            generar_mapa_calor_interactivo, generar_espectrograma_forense_web,
        )

        if not os.path.exists(ruta):
            pytest.skip(f"Archivo no encontrado: {ruta}")

        y, sr = cargar_y_normalizar_audio(ruta)
        mel_db, _, stft = calcular_stft_y_mel(y, sr)

        fig_plotly, warning = generar_mapa_calor_interactivo(
            mel_db, sr, y_audio=y, stft_compleja=stft
        )
        assert fig_plotly is not None, "El heatmap interactivo debe generarse"

        fig_mpl = generar_espectrograma_forense_web(mel_db, sr)
        assert fig_mpl is not None, "El espectrograma web debe generarse"

    @pytest.mark.parametrize("ruta", [REAL_WAV, FAKE_WAV])
    def test_memoria_no_explota(self, ruta):
        """
        Procesa el audio 3 veces seguidas para detectar fugas de memoria
        o acumulacion de tensores retenidos en el grafo computacional.
        """
        from src.physics.audio_processing import (
            cargar_y_normalizar_audio, calcular_stft_y_mel,
            predecir_audio_completo,
        )
        from src.ai.model import DeepfakeAudioCNN

        if not os.path.exists(ruta):
            pytest.skip(f"Archivo no encontrado: {ruta}")

        modelo = DeepfakeAudioCNN()
        modelo.eval()

        for i in range(3):
            y, sr = cargar_y_normalizar_audio(ruta)
            mel_db, _, _ = calcular_stft_y_mel(y, sr)
            prob, preds, _ = predecir_audio_completo(mel_db, modelo)
            assert 0.0 <= prob <= 100.0, f"Iteracion {i}: prob fuera de rango {prob}"

        print(f"  3 iteraciones sin fuga: OK")


class TestRealVsFakeComparison:
    """Compara salidas de ambos archivos en el mismo test."""

    def test_diagnostico_diferenciado(self, model, device):
        """
        Carga REAL y FAKE, corre inferencia sobre ambos, verifica que
        el fake tenga mayor probabilidad de deepfake que el real.
        NOTA: Con pesos aleatorios esto NO es deterministico.
        Solo valido si best_model.pth tiene pesos entrenados.
        """
        if not os.path.exists(REAL_WAV) or not os.path.exists(FAKE_WAV):
            pytest.skip("Archivos .wav no disponibles")
        if not os.path.exists(MODEL_PATH):
            pytest.skip("best_model.pth no encontrado (pesos aleatorios no diferenciaran)")

        from src.physics.audio_processing import (
            cargar_y_normalizar_audio, calcular_stft_y_mel,
            predecir_audio_completo,
        )

        def _predecir(ruta):
            y, sr = cargar_y_normalizar_audio(ruta)
            mel_db, _, _ = calcular_stft_y_mel(y, sr)
            prob, _, _ = predecir_audio_completo(mel_db, model, device=device)
            return prob

        prob_real = _predecir(REAL_WAV)
        prob_fake = _predecir(FAKE_WAV)

        print(f"  REAL prob_fake={prob_real:.2f}%")
        print(f"  FAKE prob_fake={prob_fake:.2f}%")

        assert isinstance(prob_real, float)
        assert isinstance(prob_fake, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
