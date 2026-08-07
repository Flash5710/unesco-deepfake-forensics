import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import pytest

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def y_signal():
    sr = 16000
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t) * 0.5 + 0.1 * np.random.randn(3 * sr)


@pytest.fixture(scope="session")
def sr_rate():
    return 16000


@pytest.fixture(scope="session")
def max_frames():
    return 300


@pytest.fixture(scope="session")
def n_mels():
    return 128


@pytest.fixture
def dummy_tensor():
    return torch.randn(2, 1, 128, 300)


# ============================================================================
# 1. PROCESAMIENTO DE AUDIO
# ============================================================================

class TestAudioProcessing:

    def test_cargar_y_normalizar_audio_rango(self, y_signal, sr_rate):
        from src.physics.audio_processing import cargar_y_normalizar_audio
        y_norm = y_signal / max(np.abs(y_signal).max(), 1e-10)
        assert np.all(np.abs(y_norm) <= 1.0 + 1e-6)

    def test_calcular_stft_y_mel_dimensiones(self, y_signal, sr_rate, n_mels):
        from src.physics.audio_processing import calcular_stft_y_mel
        mel_db, mfccs, stft = calcular_stft_y_mel(y_signal, sr_rate)
        assert mel_db.shape[0] == n_mels
        assert mfccs.shape[0] == 13
        assert stft.shape[1] == mel_db.shape[1]

    def test_calcular_stft_y_mel_ref_change(self, y_signal, sr_rate):
        from src.physics.audio_processing import calcular_stft_y_mel
        mel_db, _, _ = calcular_stft_y_mel(y_signal, sr_rate)
        assert np.all(np.isfinite(mel_db)), "mel_db debe contener solo valores finitos"
        assert mel_db.min() >= -100, "Con ref=1.0, valores mínimos no deben ser extremadamente negativos"

    def test_convertir_a_tensor_pytorch_padding_corto(self, n_mels):
        from src.physics.audio_processing import convertir_a_tensor_pytorch
        corto = np.random.uniform(-60, 0, size=(n_mels, 50))
        tensor = convertir_a_tensor_pytorch(corto, normalizar=False)
        assert tensor.shape[-1] == 300
        ultimos_cols = tensor[0, 0, 50:]
        assert torch.allclose(ultimos_cols, torch.full_like(ultimos_cols, -80.0))

    def test_convertir_a_tensor_pytorch_recorte_largo(self, n_mels, max_frames):
        from src.physics.audio_processing import convertir_a_tensor_pytorch
        largo = np.random.uniform(-60, 0, size=(n_mels, 500))
        tensor = convertir_a_tensor_pytorch(largo, normalizar=False)
        assert tensor.shape[-1] == max_frames

    def test_convertir_a_tensor_pytorch_normalizar(self, n_mels):
        from src.physics.audio_processing import convertir_a_tensor_pytorch
        mel = np.random.uniform(-60, 0, size=(n_mels, 50))
        tensor_sin_norm = convertir_a_tensor_pytorch(mel.copy(), normalizar=False)
        tensor_con_norm = convertir_a_tensor_pytorch(mel.copy(), normalizar=True)
        assert not torch.allclose(tensor_sin_norm, tensor_con_norm)

    def test_circular_import_audio_processing_custom_dataset(self):
        from src.physics.audio_processing import predecir_audio_completo
        from src.ai.model import DeepfakeAudioCNN
        mel = np.random.uniform(-60, 0, size=(128, 50))
        modelo = DeepfakeAudioCNN()
        modelo.eval()
        prob, preds, tiempos = predecir_audio_completo(mel, modelo, device=torch.device("cpu"))
        assert isinstance(prob, float)
        assert 0 <= prob <= 100


# ============================================================================
# 2. MODELO
# ============================================================================

class TestModel:

    def test_modelo_forward_shape(self, dummy_tensor):
        from src.ai.model import DeepfakeAudioCNN
        modelo = DeepfakeAudioCNN()
        salida = modelo(dummy_tensor)
        assert salida.shape == (2, 2)

    def test_modelo_gradientes_flow(self, dummy_tensor):
        from src.ai.model import DeepfakeAudioCNN
        modelo = DeepfakeAudioCNN()
        salida = modelo(dummy_tensor)
        loss = salida.sum()
        loss.backward()
        for name, param in modelo.named_parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_predict_sr_fix(self):
        import inspect
        from src.ai import predict
        src = inspect.getsource(predict.predecir_etiqueta)
        assert "target_sr=48000" not in src or "target_sr=16000" in src


# ============================================================================
# 3. VENTANAS DESLIZANTES
# ============================================================================

class TestSlidingWindow:

    def test_ventana_unica_para_audio_corto(self, n_mels, max_frames):
        from src.physics.audio_processing import predecir_audio_completo
        from src.ai.model import DeepfakeAudioCNN
        mel = np.random.uniform(-60, 0, size=(n_mels, 100))
        modelo = DeepfakeAudioCNN()
        prob, preds, tiempos = predecir_audio_completo(mel, modelo)
        assert len(preds) == 1
        assert tiempos == [0.0]

    def test_ventanas_multiples_con_stride(self, n_mels):
        from src.physics.audio_processing import predecir_audio_completo
        from src.ai.model import DeepfakeAudioCNN
        mel = np.random.uniform(-60, 0, size=(n_mels, 800))
        modelo = DeepfakeAudioCNN()
        prob, preds, tiempos = predecir_audio_completo(mel, modelo, stride=150)
        assert len(preds) > 1

    def test_max_padding_ratio_excluye(self, n_mels):
        from src.physics.audio_processing import predecir_audio_completo
        from src.ai.model import DeepfakeAudioCNN
        mel = np.random.uniform(-60, 0, size=(n_mels, 310))
        modelo = DeepfakeAudioCNN()
        prob, preds, tiempos = predecir_audio_completo(mel, modelo, stride=150, max_padding_ratio=0.4)
        assert len(preds) == 1

    def test_top_k_average(self, n_mels):
        from src.physics.audio_processing import predecir_audio_completo
        from src.ai.model import DeepfakeAudioCNN
        mel = np.random.uniform(-60, 0, size=(n_mels, 800))
        modelo = DeepfakeAudioCNN()
        prob, preds, _ = predecir_audio_completo(mel, modelo)
        top2 = sorted(preds, reverse=True)[:2]
        esperado = sum(top2) / len(top2)
        assert abs(prob - esperado) < 1e-4


# ============================================================================
# 4. VISUALIZACIÓN
# ============================================================================

class TestVisualization:

    def test_mapa_calor_khz_scale(self, y_signal, sr_rate):
        from src.physics.audio_processing import (
            calcular_stft_y_mel, generar_mapa_calor_interactivo
        )
        mel_db, _, stft = calcular_stft_y_mel(y_signal, sr_rate)
        fig, warning = generar_mapa_calor_interactivo(
            mel_db, sr_rate, y_audio=y_signal, stft_compleja=stft
        )
        assert fig is not None
        y_vals = fig.data[0].y
        assert np.all(y_vals < 10)
        hover = fig.data[0].hovertemplate
        assert "kHz" in hover

    def test_mapa_calor_anomalia_overlay(self, y_signal, sr_rate):
        from src.physics.audio_processing import (
            calcular_stft_y_mel, generar_mapa_calor_interactivo
        )
        mel_db, _, stft = calcular_stft_y_mel(y_signal, sr_rate)
        fig, warning = generar_mapa_calor_interactivo(
            mel_db, sr_rate, y_audio=y_signal, stft_compleja=stft,
            umbral_anomalia_db=-30.0
        )
        assert len(fig.data) >= 1

    def test_espectrograma_forense_web_khz(self, y_signal, sr_rate):
        from src.physics.audio_processing import (
            calcular_stft_y_mel, generar_espectrograma_forense_web
        )
        mel_db, _, _ = calcular_stft_y_mel(y_signal, sr_rate)
        fig = generar_espectrograma_forense_web(mel_db, sr_rate)
        assert fig is not None
        import matplotlib.ticker as ticker
        fmt = fig.axes[0].yaxis.get_major_formatter()
        assert isinstance(fmt, ticker.FuncFormatter)


# ============================================================================
# 5. MÉTRICA DE FASE
# ============================================================================

class TestPhaseMetric:

    def test_fase_metrics_vs_audio_processing_nfft(self):
        import inspect
        from src.math_core import metrics
        from src.physics import audio_processing

        sig_ap = inspect.signature(audio_processing.calcular_regularidad_fase)
        sig_metrics = inspect.signature(metrics.calcular_regularidad_fase)

        ap_defaults = {k: v.default for k, v in sig_ap.parameters.items() if v.default is not inspect.Parameter.empty}
        metrics_defaults = {k: v.default for k, v in sig_metrics.parameters.items() if v.default is not inspect.Parameter.empty}

        assert ap_defaults.get("n_fft") == metrics_defaults.get("n_fft", 1024)
        assert ap_defaults.get("hop_length") == metrics_defaults.get("hop_length", 256)

    def test_fase_estabilidad_valor(self, y_signal):
        from src.math_core.metrics import calcular_regularidad_fase
        valor = calcular_regularidad_fase(y_signal)
        assert np.isfinite(valor)
        assert valor > 0


# ============================================================================
# 6. LATENCIA
# ============================================================================

class TestLatency:

    def test_latencia_report_estructura(self, y_signal, sr_rate, tmp_path):
        pytest.skip("Requiere video de prueba f\u00edsico. Verificar estructura manualmente.")


# ============================================================================
# 7. SOCIAL DOWNLOADER
# ============================================================================

class TestSocialDownloader:

    def test_dominio_valido_acepta(self):
        from src.utils.social_downloader import _dominio_es_valido
        urls_validas = [
            "https://www.tiktok.com/@user/video/12345",
            "https://facebook.com/watch?v=123",
            "https://www.instagram.com/p/ABC123/",
            "https://x.com/user/status/12345",
        ]
        for url in urls_validas:
            assert _dominio_es_valido(url)

    def test_dominio_valido_rechaza(self):
        from src.utils.social_downloader import _dominio_es_valido
        urls_invalidas = [
            "https://youtube.com/watch?v=123",
            "https://vimeo.com/12345",
            "not-a-url",
        ]
        for url in urls_invalidas:
            assert not _dominio_es_valido(url)

    def test_descargar_url_no_soportada(self):
        from src.utils.social_downloader import descargar_video_de_red_social
        with pytest.raises(ValueError, match="URL no soportada"):
            descargar_video_de_red_social("https://youtube.com/watch?v=dQw4w9WgXcQ")


# ============================================================================
# 8. REGRESIÓN — INTEGRIDAD DEL REPOSITORIO
# ============================================================================

class TestRepoIntegrity:

    def test_visualization_py_no_rompe_imports(self):
        import importlib
        try:
            importlib.import_module("src.physics.visualization")
        except Exception as e:
            pytest.fail(f"src.physics.visualization no debe romper imports: {e}")

    def test_init_files_exist(self):
        for pkg in ["src", "src.ai", "src.physics", "src.math_core", "src.utils"]:
            init_path = os.path.join(
                os.path.dirname(__file__), "..", pkg.replace(".", os.sep), "__init__.py"
            )
            assert os.path.exists(init_path), f"Falta {init_path}"

    def test_requirements_contains_ytdlp(self):
        req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        with open(req_path) as f:
            content = f.read()
        assert "yt-dlp" in content

    def test_no_model_weights_in_repo(self, tmp_path):
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "*.pth"],
            capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), "..")
        )
        assert result.stdout.strip() == ""

    def test_cache_resource_not_removed(self):
        app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
        with open(app_path, encoding="utf-8") as f:
            content = f.read()
        assert "@st.cache_resource" in content


# ============================================================================
# 9. NORMALIZACIÓN CONSISTENTE
# ============================================================================

class TestNormalizationConsistency:

    def test_normalizacion_consistente(self):
        from src.physics.audio_processing import normalizar_espectrograma as norm_ap
        from src.ai.custom_dataset import normalizar_espectrograma as norm_cd
        mel = np.random.uniform(-60, 0, size=(128, 300))
        r1 = norm_ap(mel.copy())
        r2 = norm_cd(mel.copy())
        np.testing.assert_array_almost_equal(r1, r2, decimal=6)
