import numpy as np


def augmentar_audio_crudo(y: np.ndarray, sr: int, intensidad: float = 1.0) -> np.ndarray:
    if intensidad <= 0.0 or len(y) < sr:
        return y

    try:
        from audiomentations import Compose, Mp3Compression, AddGaussianNoise, PitchShift, TimeStretch

        i = min(max(intensidad, 0.0), 1.0)
        transformador = Compose([
            Mp3Compression(min_bitrate=32, max_bitrate=128, p=0.5 * i),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.4 * i),
            PitchShift(min_semitones=-1, max_semitones=1, p=0.3 * i),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.2 * i),
        ])
        y_aug = transformador(samples=y, sample_rate=sr)
        return y_aug.astype(np.float32)
    except Exception:
        return y