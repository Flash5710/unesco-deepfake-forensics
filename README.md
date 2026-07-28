<div align="center">
  <br>
  <h1>🕵️ DeepForensic</h1>
  <p><strong>Detector Forense de Deepfakes de Audio</strong></p>
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

## 📋 Descripción

**DeepForensic** es una plataforma forense multiplataforma diseñada para detectar **deepfakes de audio** mediante un enfoque híbrido que combina **análisis físico de señales**, **métricas matemáticas de fase** e **inteligencia artificial profunda** (CNN).

Desarrollada para el **UNESCO Youth Hackathon 2026**, la herramienta permite a periodistas, investigadores forenses y público general verificar la autenticidad de contenido auditivo proveniente de redes sociales, archivos locales o enlaces directos.

La detección se basa en un principio fundamental: **los vocoders neuronales (WaveNet, Tacotron, etc.) dejan artefactos espectrales y discontinuidades de fase que no ocurren en la voz humana natural**. DeepForensic explota estas huellas forenses para clasificar audios como REAL o FAKE con alta precisión.

---

## ✨ Características Clave

| Característica | Descripción |
|---|---|
| **🔬 Análisis Dual** | Combina métricas físicas (fase, energía espectral) con una CNN profunda para máxima precisión |
| **🌐 Multi-plataforma** | Interfaz web (Streamlit), API REST (FastAPI) y extensión para Chrome |
| **📱 Redes Sociales** | Descarga y analiza videos directamente desde YouTube, Twitter/X, TikTok, Instagram y Facebook |
| **🗺️ Heatmap Interactivo** | Visualización forense con detección automática de anomalías espectrales y de fase |
| **🔍 Explicabilidad (Grad-CAM)** | Mapas de activación que revelan qué regiones del espectro llevaron al veredicto |
| **📄 Reportes PDF** | Genera reportes forenses descargables con evidencias visuales y métricas |
| **🌍 Internacionalización** | Interfaz completa en Español e Inglés |
| **🔊 Accesibilidad** | Síntesis de voz del resultado forense (Web Speech API) |
| **📊 Análisis Temporal** | Ventanas deslizantes sobre el audio completo, sin truncamiento |
| **⚡ Baja Latencia** | Pipeline optimizado con precomputación en GPU y caching de tensores |

---

## 🏗️ Arquitectura del Sistema

```
                    ┌─────────────────────────────────────────────────────┐
                    │               ENTRADA DE DATOS                      │
                    │  [Archivo local]  [URL red social]  [Página web]    │
                    └──────────────────────┬──────────────────────────────┘
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │          EXTRACCIÓN DE AUDIO (FFmpeg)               │
                    │  • Monofonización   • 16 kHz   • Normalización pico │
                    └──────────────────────┬──────────────────────────────┘
                                           ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │               PIPELINE DE PROCESAMIENTO HÍBRIDO                  │
        │                                                                  │
        │  ┌─────────────────────┐        ┌────────────────────────────┐   │
        │  │   MÓDULO FÍSICO     │        │    MÓDULO DE IA (CNN)       │   │
        │  │                     │        │                             │   │
        │  │ • STFT (n_fft=2048) │        │ • Ventanas deslizantes     │   │
        │  │ • Mel (128 bandas)  │        │   (300 frames, stride 150) │   │
        │  │ • MFCCs (13 coefs)  │        │ • Normalización individual │   │
        │  │ • Filtro Butterworth│        │   por ventana (z-score)    │   │
        │  │   paso-alto (4 kHz) │        │ • Padding con -80 dB       │   │
        │  │ • Regularidad de    │        │ • DeepfakeAudioCNN         │   │
        │  │   fase (varianza    │        │   (4 bloques conv + ADAPT) │   │
        │  │   de derivada)      │        │ • Grad-CAM (capa conv4)    │   │
        │  └──────────┬──────────┘        └──────────────┬─────────────┘   │
        │             │                                   │                 │
        └─────────────┼───────────────────────────────────┼─────────────────┘
                      ▼                                   ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                    VEREDICTO HÍBRIDO                             │
        │  • Top-2 promedio de ventanas más deepfake                     │
        │  • Umbral: >50% → DEEPFAKE, ≤50% → REAL                       │
        │  • Inestabilidad de fase > 3.5 → alerta de manipulación        │
        └──────────────────────────┬───────────────────────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                    VISUALIZACIÓN Y REPORTES                      │
        │  • Heatmap interactivo (Plotly) con anomalías resaltadas        │
        │  • Métricas de predicción con códigos de color                 │
        │  • Grad-CAM: regiones críticas del espectro                    │
        │  • Reporte PDF descargable (reportlab)                         │
        │  • Síntesis de voz del resultado                               │
        └──────────────────────────────────────────────────────────────────┘
```

---

## 🧪 La Ciencia Detrás de la Detección

### 🔷 Análisis de Fase (La "Huella Digital" del Vocoder)

Los vocoders neuronales generan audio muestreando una señal continua a partir de representaciones paramétricas. Este proceso introduce **discontinuidades en la fase** de la señal que son estadísticamente detectables:

1. Se calcula la **STFT compleja** de la señal
2. Se extrae el **ángulo (fase)** en radianes: `φ(t, f) = arctan2(imag, real)`
3. Se calcula la **derivada temporal**: `Δφ(t, f) = φ(t+1, f) − φ(t, f)`
4. Se **envuelve** a `[−π, π]` para eliminar saltos naturales de `2π`
5. La **varianza** de `Δφ` a lo largo del tiempo mide qué tan caótica es la fase

**Interpretación**: La voz humana natural tiene una fase suave y predecible. Un valor de **inestabilidad > 3.5** sugiere fuertemente síntesis artificial.

### 🔷 Mel-Spectrogram y Psicoacústica

El oído humano percibe la frecuencia de forma logarítmica. La **escala Mel** modela esta percepción:

```
mel(f) = 2595 · log₁₀(1 + f/700)
```

Transformamos el audio a un espectrograma de 128 bandas Mel en dB, que sirve como entrada para la CNN. Esto reduce la dimensionalidad mientras preserva la información perceptual relevante.

### 🔷 Filtro Paso-Alto (4 kHz)

Los artefactos de vocoder tienden a concentrarse en **altas frecuencias** (> 4 kHz), donde el habla humana tiene menos energía estructural. Aplicamos un filtro Butterworth de orden 5 (fase cero, orden efectivo 10) con `scipy.signal.filtfilt()` para aislar estas regiones sospechosas.

### 🔷 Ventanas Deslizantes con Top-2 Averaging

En lugar de truncar el audio, lo dividimos en **ventanas de ~9.6 segundos** (300 frames) con stride de 150 frames. Cada ventana se normaliza individualmente (z-score sobre su contenido real, excluyendo el padding de -80 dB). El **veredicto final** es el promedio de las dos ventanas con mayor probabilidad deepfake, garantizando que el análisis cubra todo el audio sin perder información.

### 🔷 DeepfakeAudioCNN

Arquitectura convolucional diseñada específicamente para clasificación de espectrogramas:

```
Input: (B, 1, 128, 300)
  Conv2D(1→32, k=3) → BatchNorm → ReLU → MaxPool(2×2)
  Conv2D(32→64, k=3) → BatchNorm → ReLU → MaxPool(2×2)
  Conv2D(64→128, k=3) → BatchNorm → ReLU → MaxPool(2×2)
  Conv2D(128→256, k=3) → BatchNorm → ReLU → MaxPool(2×2)
  AdaptiveAvgPool2d(4×4)    ← compatible con cualquier duración
  FC(4096→256) → Dropout(0.3) → FC(256→2) → Softmax
Output: [P(REAL), P(FAKE)]
```

**Clave**: El `AdaptiveAvgPool2d` permite procesar audios de cualquier duración sin distorsión.

### 🔷 Grad-CAM (Explicabilidad)

Para entender *qué* llevó al modelo a decidir, aplicamos **Grad-CAM** sobre la última capa convolucional (`conv4`):

1. Propagamos el audio hacia adelante
2. Calculamos el gradiente de la clase objetivo respecto a las activaciones de `conv4`
3. Pesamos los mapas de activación por el gradiente promedio (`GAP`)
4. Aplicamos ReLU para retener solo las regiones con influencia positiva
5. Interpolamos al tamaño del espectrograma original

El resultado es un **mapa de calor** que muestra las regiones temporales y frecuenciales más determinantes para el veredicto.

---

## 📊 Rendimiento del Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | ≥ 97% |
| **F1-Score (Fake)** | **97.65%** |
| **Recall (Fake)** | **96.47%** |
| **Arquitectura** | DeepfakeAudioCNN (4 bloques conv) |
| **Parámetros** | ~500K |
| **Latencia media** | < 100 ms por ventana (CPU) |
| **Formato entrada** | Mel-spectrogram (128×300) |

---

## 📦 Requisitos e Instalación

### Prerrequisitos
- **Python 3.12+**
- **ffmpeg** (incluido automáticamente vía `imageio-ffmpeg`)
- **Git**

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/unesco-deepfake-forensics.git
cd unesco-deepfake-forensics

# 2. Crear y activar entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota sobre GPU (CUDA)**: Si tienes una GPU NVIDIA, instala PyTorch con soporte CUDA:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 🚀 Cómo Ejecutar

### 🔹 Modo 1: Aplicación Web (Streamlit)

```bash
streamlit run app.py
```
Abre en [http://localhost:8501](http://localhost:8501)

Interfaz completa para subir archivos, pegar enlaces de redes sociales y visualizar resultados forenses con mapas de calor interactivos y Grad-CAM.

### 🔹 Modo 2: API REST (para Extensión Chrome)

```bash
# Desde la raíz del proyecto
uvicorn api_extension_workspace.servidor_api:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en [http://localhost:8000](http://localhost:8000) con los siguientes endpoints:

| Endpoint | Método | Descripción |
|---|---|---|
| `/analyze_file` | POST | Analiza un archivo de audio/video subido |
| `/analyze_url` | POST | Analiza contenido desde una URL de red social |
| `/health` | GET | Health check del servidor |

Documentación interactiva: [http://localhost:8000/docs](http://localhost:8000/docs)

### 🔹 Modo 3: Extensión de Chrome

1. Asegúrate de que la **API REST** esté corriendo (Modo 2)
2. Abre Chrome y ve a `chrome://extensions/`
3. Activa el **Modo desarrollador** (esquina superior derecha)
4. Haz clic en **"Cargar extensión sin empaquetar"**
5. Selecciona la carpeta `api_extension_workspace/chrome_extension/`
6. El icono de DeepForensic aparecerá en la barra de extensiones

**Funcionalidades de la extensión:**
- **FAB flotante** en YouTube, Twitter/X, Instagram, TikTok y Facebook: haz clic en 🔍 para analizar la página actual
- **Menú contextual**: haz clic derecho sobre cualquier enlace para analizarlo
- **Popup**: sube archivos, pega URLs, revisa historial (últimos 100 análisis)
- **Notificaciones**: recibirás notificaciones con el veredicto al analizar desde el menú contextual
- **Historial persistente** con indicadores visuales (rojo = deepfake, verde = real)

---

## 🗂️ Estructura del Proyecto

```
unesco-deepfake-forensics/
│
├── app.py                         # Frontend Streamlit (interfaz web principal)
├── requirements.txt               # Dependencias del proyecto
├── test-integracion.py            # Prueba de integración manual
│
├── src/
│   ├── sanatizar_dataset.py       # Limpieza de archivos corruptos
│   │
│   ├── math_core/                 # 🌐 Módulo matemático
│   │   ├── metrics.py             # Regularidad de fase (núcleo forense)
│   │   └── normalization.py       # Normalización de audio y espectrogramas
│   │
│   ├── physics/                   # ⚛️ Procesamiento de señales
│   │   └── audio_processing.py    # Pipeline físico completo (STFT, Mel, fase, heatmap)
│   │
│   ├── ai/                        # 🧠 Inteligencia Artificial
│   │   ├── model.py               # DeepfakeAudioCNN (arquitectura PyTorch)
│   │   ├── inference.py           # Motor de inferencia rápida
│   │   ├── custom_dataset.py      # Dataset con cache y augmentación
│   │   ├── train.py               # Entrenamiento con GPU + early stopping
│   │   ├── audio_augmentation.py  # Aumento de datos acústico (MP3, ruido, pitch)
│   │   ├── gradcam.py             # Explicabilidad: Grad-CAM
│   │   ├── predict.py             # Predicción desde archivo/tensor
│   │   ├── pipeline.py            # Pipeline de integración IA
│   │   ├── evaluate.py            # Evaluación comparativa de modelos
│   │   ├── download_dataset.py    # Datos mock para desarrollo
│   │   ├── preparar_datos.py      # Descarga datasets Kaggle
│   │   ├── preparar_fakeavceleb.py # Prepara dataset FakeAVCeleb
│   │   ├── split_datos.py         # División train/test
│   │   ├── stress_test.py         # Prueba de estrés de inferencia
│   │   ├── ui_hooks.py            # Hook de inicialización para UI
│   │   ├── best_model.pth         # Pesos v1 (original)
│   │   ├── best_model_v2_augmented.pth  # Pesos v2 (con augmentación)
│   │   └── best_model_wav.pth     # Pesos pre-entrenados para fine-tuning
│   │
│   └── utils/                     # 🛠️ Utilidades
│       ├── social_downloader.py   # Descarga de redes sociales (yt-dlp)
│       ├── pdf_generator.py       # Generación de reportes PDF (reportlab)
│       └── i18n.py                # Motor de internacionalización ES/EN
│
├── api_extension_workspace/       # 🌐 API y Extensión Chrome
│   ├── servidor_api.py            # API REST (FastAPI)
│   └── chrome_extension/          # Extensión de Chrome (MV3)
│       ├── manifest.json          # Permisos y configuración
│       ├── background.js          # Service worker (menús, notificaciones)
│       ├── content.js             # Content script (FAB + modal flotante)
│       ├── popup.html             # Interfaz del popup
│       ├── popup.js               # Lógica del popup
│       ├── popup.css              # Estilos dark mode
│       ├── i18n.js                # Traducciones frontend
│       └── icon*.png              # Iconos de la extensión
│
├── i18n/
│   └── strings.json               # Strings de traducción ES/EN
│
├── data/                          # 📁 Datos
│   ├── train/{real,fake}/         # Audios de entrenamiento
│   ├── test/{real,fake}/          # Audios de prueba
│   ├── samples/                   # Audios de muestra
│   └── cache/                     # Caché de tensores precomputados
│
└── tests/                         # ✅ Tests
    ├── test_integration.py        # Tests end-to-end
    └── test_regression.py         # Tests de regresión
```

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnología | Propósito |
|---|---|---|
| **Lenguaje** | Python 3.12+ | Núcleo del proyecto |
| **Deep Learning** | PyTorch, torchaudio | CNN, Grad-CAM, transformadas GPU |
| **Señales** | librosa, scipy, numpy | STFT, Mel, MFCCs, filtros Butterworth |
| **Frontend** | Streamlit | Dashboard web interactivo |
| **Visualización** | Plotly, Matplotlib | Heatmaps, espectrogramas |
| **API** | FastAPI, Uvicorn | REST API para la extensión |
| **Extensión** | Chrome MV3, JavaScript | Content script, popup, service worker |
| **Audio/Video** | FFmpeg, imageio-ffmpeg | Extracción de pista de audio |
| **Descargas** | yt-dlp | Descarga de redes sociales |
| **PDF** | reportlab | Reportes forenses |
| **Datasets** | kagglehub | Descarga de datasets |
| **Testing** | pytest | Tests de integración y regresión |

---

## 📚 Datasets

DeepForensic se entrena con los siguientes datasets, descargables automáticamente:

| Dataset | Clase | Fuente |
|---|---|---|
| **FakeAudio** | FAKE | Kaggle - `walimuhammadahmad/fakeaudio` |
| **Speaker Recognition** | REAL | Kaggle - `vjcalling/speaker-recognition-audio-dataset` |
| **FakeAVCeleb** | FAKE + REAL | Kaggle - `shreyaty08/fakeavceleb` |

Para preparar los datos:
```bash
# Descargar y submuestrear 5000 audios por clase
python src/ai/preparar_datos.py

# (Opcional) Preparar FakeAVCeleb
python src/ai/preparar_fakeavceleb.py

# Dividir en train/test (80/20)
python src/ai/split_datos.py

# Sanitizar archivos corruptos
python src/ai/../sanatizar_dataset.py
```

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Tests específicos
pytest tests/test_integration.py -v
pytest tests/test_regression.py -v
```

---

## 🏋️ Entrenar el Modelo

```bash
python src/ai/train.py
```

El script:
1. Carga los datos desde `data/train/`
2. Precomputa los waveforms crudos en RAM
3. Divide automáticamente en train/val (85/15)
4. Aplica aumento de datos (acústico + espectrograma)
5. Entrena con early stopping y reducción de learning rate
6. Evalúa en el conjunto de test
7. Guarda el mejor modelo en `src/ai/best_model_v2_augmented.pth`

---

## 📄 Licencia

Este proyecto fue desarrollado para el **UNESCO Youth Hackathon 2026**. Todos los derechos reservados a sus autores.

---

<div align="center">
  <p>
    <strong>DeepForensic</strong> — Por un ecosistema digital más seguro y verificado.
  </p>
  <p>
    <em>"La verdad no teme la investigación"</em>
  </p>
</div>