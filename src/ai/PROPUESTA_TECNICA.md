# Propuesta Técnica: Módulo de Inteligencia Artificial para la Detección de Deepfakes de Audio
**Organización / Competencia:** UNESCO Hackathon  



## 1. Resumen Ejecutivo
El presente módulo implementa un pipeline de aprendizaje profundo (*Deep Learning*) diseñado para clasificar audios sospechosos entre clases **REAL** y **FAKE**. La solución se optimizó para ofrecer alta precisión y baja latencia, integrándose de manera transparente con las etapas de procesamiento físico y matemático de señales.



## 2. Arquitectura de la Red Neuronal (CNN Ligera)
Se diseñó la arquitectura `DeepfakeAudioCNN` en PyTorch, optimizada para procesar espectrogramas de Mel:
* **Bloques Convolucionales:** Dos capas de convolución 2D  acompañadas de normalización por lotes  y funciones de activación ReLU para la extracción eficiente de características espectro-temporales.
* **Reducción Adaptativa:** Incorporación de `AdaptiveAvgPool2d((4, 4))` para garantizar compatibilidad con audios de cualquier duración temporal sin distorsionar la resolución de frecuencia.
* **Regularización y Clasificación:** Inclusión de capas `Dropout(0.3)` para mitigar el sobreajuste (*overfitting*) y capas lineales finales para la proyección a dos clases.



## 3. Pipeline de Entrenamiento y Aumento de Datos
* **Robusteza mediante Ruido:** Inyección de ruido blanco gaussiano aleatorio (`torch.randn_like`) durante la fase de carga de datos (`custom_dataset.py`) para simular variaciones de señal del mundo real.
* **Función de Pérdida y Optimización:** Uso de `CrossEntropyLoss` junto con el optimizador `Adam` ($lr=0.001$).


## 4. Integración y Módulo de Inferencia
* **Inferencia en Tiempo Real:** Implementación de `predict.py` e `inference.py` ejecutando `model.eval()` y desactivación de gradientes (`torch.no_grad()`) para minimizar la latencia.
* **Interfaz con la UI:** Exportación de funciones de inicialización mediante `ui_hooks.py` estructurando respuestas claras con porcentajes de confianza.