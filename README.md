<div align="center">
  <br>
  <h1>🕵️ DeepForensic</h1>
  <p><strong>Audio Deepfake Forensic Detector</strong></p>
  <p><em>UNESCO Youth Hackathon 2026</em></p>
  <br>
  <p>
    <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?logo=streamlit" alt="Streamlit">
    <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/Chrome%20Extension-MV3-4285F4?logo=googlechrome" alt="Chrome Extension">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>
  <br>
</div>

---

## 📋 Description

**DeepForensic** is a cross-platform forensic platform designed to detect **audio deepfakes** through a hybrid approach combining **physical signal analysis**, **mathematical phase metrics**, and **deep artificial intelligence** (CNN).

Developed for the **UNESCO Youth Hackathon 2026**, the tool enables journalists, forensic investigators, and the general public to verify the authenticity of audio content from social media, local files, or direct links.

Detection is based on a fundamental principle: **neural vocoders (WaveNet, Tacotron, etc.) leave spectral artifacts and phase discontinuities that do not occur in natural human speech**. DeepForensic exploits these forensic fingerprints to classify audio as REAL or FAKE with high accuracy.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **🔬 Dual Analysis** | Combines physical metrics (phase, spectral energy) with a deep CNN for maximum accuracy |
| **🌐 Cross-platform** | Web interface (Streamlit), REST API (FastAPI), and Chrome extension |
| **📱 Social Media** | Download and analyze videos directly from YouTube, Twitter/X, TikTok, Instagram, and Facebook |
| **🗺️ Interactive Heatmap** | Forensic visualization with automatic detection of spectral and phase anomalies |
| **🔍 Explainability (Grad-CAM)** | Activation maps revealing which spectral regions drove the verdict |
| **📄 PDF Reports** | Generates downloadable forensic reports with visual evidence and metrics |
| **🌍 Internationalization** | Full interface in English and Spanish |
| **🔊 Accessibility** | Speech synthesis of the forensic result (Web Speech API) |
| **📊 Temporal Analysis** | Sliding windows over the entire audio, no truncation |
| **⚡ Low Latency** | Optimized pipeline with GPU precomputation and tensor caching |

---

## 🏗️ System Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                 DATA INPUT                          │
                    │  [Local file]  [Social URL]  [Web page]            │
                    └──────────────────────┬──────────────────────────────┘
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │          AUDIO EXTRACTION (FFmpeg)                  │
                    │  • Mono conversion  • 16 kHz  • Peak normalization │
                    └──────────────────────┬──────────────────────────────┘
                                           ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │               HYBRID PROCESSING PIPELINE                        │
        │                                                                  │
        │  ┌─────────────────────┐        ┌────────────────────────────┐   │
        │  │   PHYSICAL MODULE   │        │     AI MODULE (CNN)        │   │
        │  │                     │        │                             │   │
        │  │ • STFT (n_fft=2048) │        │ • Sliding windows          │   │
        │  │ • Mel (128 bands)   │        │   (300 frames, stride 150) │   │
        │  │ • MFCCs (13 coefs)  │        │ • Per-window normalization │   │
        │  │ • Butterworth       │        │   (z-score)                │   │
        │  │   high-pass (4 kHz) │        │ • Padding with -80 dB      │   │
        │  │ • Phase regularity  │        │ • DeepfakeAudioCNN         │   │
        │  │   (derivative       │        │   (4 conv blocks + ADAPT)  │   │
        │  │    variance)        │        │ • Grad-CAM (conv4 layer)   │   │
        │  └──────────┬──────────┘        └──────────────┬─────────────┘   │
        │             │                                   │                 │
        └─────────────┼───────────────────────────────────┼─────────────────┘
                      ▼                                   ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                    HYBRID VERDICT                                │
        │  • Top-2 average of most deepfake windows                       │
        │  • Threshold: >50% → DEEPFAKE, ≤50% → REAL                     │
        │  • Phase instability > 3.5 → manipulation alert                 │
        └──────────────────────────┬───────────────────────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                    VISUALIZATION AND REPORTS                     │
        │  • Interactive heatmap (Plotly) with highlighted anomalies       │
        │  • Prediction metrics with color codes                          │
        │  • Grad-CAM: critical spectral regions                          │
        │  • Downloadable PDF report (reportlab)                          │
        │  • Speech synthesis of the result                               │
        └──────────────────────────────────────────────────────────────────┘
```

---

## 🧪 The Science Behind Detection

### 🔷 Phase Analysis (The Vocoder's "Digital Fingerprint")

Neural vocoders generate audio by sampling a continuous signal from parametric representations. This process introduces **phase discontinuities** in the signal that are statistically detectable:

1. Compute the **complex STFT** of the signal
2. Extract the **angle (phase)** in radians: `φ(t, f) = arctan2(imag, real)`
3. Compute the **temporal derivative**: `Δφ(t, f) = φ(t+1, f) − φ(t, f)`
4. **Wrap** to `[−π, π]` to remove natural `2π` jumps
5. The **variance** of `Δφ` over time measures how chaotic the phase is

**Interpretation**: Natural human speech has a smooth and predictable phase. An **instability value > 3.5** strongly suggests artificial synthesis.

### 🔷 Mel-Spectrogram and Psychoacoustics

The human ear perceives frequency logarithmically. The **Mel scale** models this perception:

```
mel(f) = 2595 · log₁₀(1 + f/700)
```

We transform the audio into a 128-band Mel spectrogram in dB, which serves as input to the CNN. This reduces dimensionality while preserving relevant perceptual information.

### 🔷 High-Pass Filter (4 kHz)

Vocoder artifacts tend to concentrate at **high frequencies** (> 4 kHz), where human speech has less structural energy. We apply a 5th-order Butterworth filter (zero-phase, effective order 10) with `scipy.signal.filtfilt()` to isolate these suspicious regions.

### 🔷 Sliding Windows with Top-2 Averaging

Instead of truncating the audio, we split it into **~9.6 second windows** (300 frames) with a stride of 150 frames. Each window is individually normalized (z-score over its real content, excluding -80 dB padding). The **final verdict** is the average of the two windows with the highest deepfake probability, ensuring the analysis covers the entire audio without losing information.

### 🔷 DeepfakeAudioCNN

Convolutional architecture specifically designed for spectrogram classification:

```
Input: (B, 1, 128, 300)
  Conv2D(1→32, k=3) → BatchNorm → ReLU → MaxPool(2×2)
  Conv2D(32→64, k=3) → BatchNorm → ReLU → MaxPool(2×2)
  Conv2D(64→128, k=3) → BatchNorm → ReLU → MaxPool(2×2)
  Conv2D(128→256, k=3) → BatchNorm → ReLU → MaxPool(2×2)
  AdaptiveAvgPool2d(4×4)    ← compatible with any duration
  FC(4096→256) → Dropout(0.3) → FC(256→2) → Softmax
Output: [P(REAL), P(FAKE)]
```

**Key**: `AdaptiveAvgPool2d` allows processing audio of any duration without distortion.

### 🔷 Grad-CAM (Explainability)

To understand *what* drove the model's decision, we apply **Grad-CAM** on the last convolutional layer (`conv4`):

1. Forward propagate the audio
2. Compute the gradient of the target class with respect to `conv4` activations
3. Weight the activation maps by the average gradient (`GAP`)
4. Apply ReLU to retain only regions with positive influence
5. Interpolate to the original spectrogram size

The result is a **heatmap** showing the temporal and frequency regions most decisive for the verdict.

---

## 📊 Model Performance

| Metric | Value |
|---------|-------|
| **Accuracy** | ≥ 97% |
| **F1-Score (Fake)** | **97.65%** |
| **Recall (Fake)** | **96.47%** |
| **Architecture** | DeepfakeAudioCNN (4 conv blocks) |
| **Parameters** | ~500K |
| **Average Latency** | < 100 ms per window (CPU) |
| **Input Format** | Mel-spectrogram (128×300) |

---

## 📦 Requirements and Installation

### Prerequisites
- **Python 3.12+**
- **ffmpeg** (automatically included via `imageio-ffmpeg`)
- **Git**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/unesco-deepfake-forensics.git
cd unesco-deepfake-forensics

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note about GPU (CUDA)**: If you have an NVIDIA GPU, install PyTorch with CUDA support:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 🚀 How to Run

### 🔹 Mode 1: Web Application (Streamlit)

```bash
streamlit run app.py
```
Opens at [http://localhost:8501](http://localhost:8501)

Full interface for uploading files, pasting social media links, and viewing forensic results with interactive heatmaps and Grad-CAM.

### 🔹 Mode 2: REST API (for Chrome Extension)

```bash
# From the project root
uvicorn api_extension_workspace.servidor_api:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000) with the following endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/analyze_file` | POST | Analyze an uploaded audio/video file |
| `/analyze_url` | POST | Analyze content from a social media URL |
| `/health` | GET | Server health check |

Interactive documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 🔹 Mode 3: Chrome Extension

1. Make sure the **REST API** is running (Mode 2)
2. Open Chrome and go to `chrome://extensions/`
3. Enable **Developer mode** (top right corner)
4. Click **"Load unpacked"**
5. Select the `api_extension_workspace/chrome_extension/` folder
6. The DeepForensic icon will appear in the extensions bar

**Extension features:**
- **Floating FAB** on YouTube, Twitter/X, Instagram, TikTok and Facebook: click 🔍 to analyze the current page
- **Context menu**: right-click any link to analyze it
- **Popup**: upload files, paste URLs, browse history (last 100 analyses)
- **Notifications**: you will receive notifications with the verdict when analyzing from the context menu
- **Persistent history** with visual indicators (red = deepfake, green = real)

---

## 🗂️ Project Structure

```
unesco-deepfake-forensics/
│
├── app.py                         # Streamlit frontend (main web interface)
├── requirements.txt               # Project dependencies
├── test-integracion.py            # Manual integration test
│
├── src/
│   ├── sanatizar_dataset.py       # Corrupt file cleanup
│   │
│   ├── math_core/                 # 🌐 Mathematical module
│   │   ├── metrics.py             # Phase regularity (forensic core)
│   │   └── normalization.py       # Audio and spectrogram normalization
│   │
│   ├── physics/                   # ⚛️ Signal processing
│   │   └── audio_processing.py    # Complete physical pipeline (STFT, Mel, phase, heatmap)
│   │
│   ├── ai/                        # 🧠 Artificial Intelligence
│   │   ├── model.py               # DeepfakeAudioCNN (PyTorch architecture)
│   │   ├── inference.py           # Fast inference engine
│   │   ├── custom_dataset.py      # Dataset with caching and augmentation
│   │   ├── train.py               # Training with GPU + early stopping
│   │   ├── audio_augmentation.py  # Acoustic data augmentation (MP3, noise, pitch)
│   │   ├── gradcam.py             # Explainability: Grad-CAM
│   │   ├── predict.py             # Prediction from file/tensor
│   │   ├── pipeline.py            # AI integration pipeline
│   │   ├── evaluate.py            # Comparative model evaluation
│   │   ├── download_dataset.py    # Mock data for development
│   │   ├── preparar_datos.py      # Download Kaggle datasets
│   │   ├── preparar_fakeavceleb.py # Prepare FakeAVCeleb dataset
│   │   ├── split_datos.py         # Train/test split
│   │   ├── stress_test.py         # Inference stress test
│   │   ├── ui_hooks.py            # UI initialization hook
│   │   ├── best_model.pth         # Weights v1 (original)
│   │   ├── best_model_v2_augmented.pth  # Weights v2 (with augmentation)
│   │   └── best_model_wav.pth     # Pre-trained weights for fine-tuning
│   │
│   └── utils/                     # 🛠️ Utilities
│       ├── social_downloader.py   # Social media downloader (yt-dlp)
│       ├── pdf_generator.py       # PDF report generation (reportlab)
│       └── i18n.py                # ES/EN internationalization engine
│
├── api_extension_workspace/       # 🌐 API and Chrome Extension
│   ├── servidor_api.py            # REST API (FastAPI)
│   └── chrome_extension/          # Chrome Extension (MV3)
│       ├── manifest.json          # Permissions and configuration
│       ├── background.js          # Service worker (menus, notifications)
│       ├── content.js             # Content script (FAB + floating modal)
│       ├── popup.html             # Popup interface
│       ├── popup.js               # Popup logic
│       ├── popup.css              # Dark mode styles
│       ├── i18n.js                # Frontend translations
│       └── icon*.png              # Extension icons
│
├── i18n/
│   └── strings.json               # ES/EN translation strings
│
├── data/                          # 📁 Data
│   ├── train/{real,fake}/         # Training audio files
│   ├── test/{real,fake}/          # Test audio files
│   ├── samples/                   # Sample audio files
│   └── cache/                     # Precomputed tensor cache
│
└── tests/                         # ✅ Tests
    ├── test_integration.py        # End-to-end tests
    └── test_regression.py         # Regression tests
```

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.12+ | Project core |
| **Deep Learning** | PyTorch, torchaudio | CNN, Grad-CAM, GPU transforms |
| **Signals** | librosa, scipy, numpy | STFT, Mel, MFCCs, Butterworth filters |
| **Frontend** | Streamlit | Interactive web dashboard |
| **Visualization** | Plotly, Matplotlib | Heatmaps, spectrograms |
| **API** | FastAPI, Uvicorn | REST API for the extension |
| **Extension** | Chrome MV3, JavaScript | Content script, popup, service worker |
| **Audio/Video** | FFmpeg, imageio-ffmpeg | Audio track extraction |
| **Downloads** | yt-dlp | Social media downloading |
| **PDF** | reportlab | Forensic reports |
| **Datasets** | kagglehub | Dataset downloading |
| **Testing** | pytest | Integration and regression tests |

---

## 📚 Datasets

DeepForensic is trained on the following datasets, automatically downloadable:

| Dataset | Class | Source |
|---|---|---|
| **FakeAudio** | FAKE | Kaggle - `walimuhammadahmad/fakeaudio` |
| **Speaker Recognition** | REAL | Kaggle - `vjcalling/speaker-recognition-audio-dataset` |
| **FakeAVCeleb** | FAKE + REAL | Kaggle - `shreyaty08/fakeavceleb` |

To prepare the data:
```bash
# Download and subsample 5000 audio files per class
python src/ai/preparar_datos.py

# (Optional) Prepare FakeAVCeleb
python src/ai/preparar_fakeavceleb.py

# Split into train/test (80/20)
python src/ai/split_datos.py

# Sanitize corrupt files
python src/ai/../sanatizar_dataset.py
```

---

## 🧪 Tests

```bash
# Run all tests
pytest tests/ -v

# Specific tests
pytest tests/test_integration.py -v
pytest tests/test_regression.py -v
```

---

## 🏋️ Train the Model

```bash
python src/ai/train.py
```

The script:
1. Loads data from `data/train/`
2. Precomputes raw waveforms in RAM
3. Automatically splits into train/val (85/15)
4. Applies data augmentation (acoustic + spectrogram)
5. Trains with early stopping and learning rate reduction
6. Evaluates on the test set
7. Saves the best model to `src/ai/best_model_v2_augmented.pth`

---

## 📄 License

This project was developed for the **UNESCO Youth Hackathon 2026**. All rights reserved to its authors.

---

<div align="center">
  <p>
    <strong>DeepForensic</strong> — For a safer, verified digital ecosystem.
  </p>
  <p>
    <em>"Truth does not fear investigation"</em>
  </p>
</div>
