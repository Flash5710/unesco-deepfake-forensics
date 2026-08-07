import base64
import io
import os
import sys
import tempfile
import uuid

import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ai.gradcam import generar_mapa_gradcam
from src.ai.inference import DeepfakeDetectorInference
from src.physics.audio_processing import (
    calcular_regularidad_fase,
    calcular_stft_y_mel,
    cargar_y_normalizar_audio,
    extraer_audio_de_video,
    predecir_audio_completo,
)
from src.utils.pdf_generator import generar_reporte_forense
from src.utils.social_downloader import descargar_video_de_red_social

app = FastAPI(title="DeepForensic API — UNESCO 2026")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "data/samples"
os.makedirs(TEMP_DIR, exist_ok=True)

class AnalyzeURLRequest(BaseModel):
    url: str

class AnalyzeResponse(BaseModel):
    veredicto: str
    confianza: float
    inestabilidad: float
    pdf_base64: str

class AnalyzeURLResponse(AnalyzeResponse):
    archivo_descargado: str

_DETECTOR: DeepfakeDetectorInference | None = None

def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = DeepfakeDetectorInference()
    return _DETECTOR

def _procesar_archivo(ruta_archivo: str, lang: str = "es") -> AnalyzeResponse:
    ext = os.path.splitext(ruta_archivo)[1].lower()
    if ext == ".mp4":
        ruta_wav = extraer_audio_de_video(ruta_archivo)
    elif ext == ".wav":
        ruta_wav = ruta_archivo
    else:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {ext}")

    y, sr = cargar_y_normalizar_audio(ruta_wav, target_sr=16000)
    mel_db, _, stft_compleja = calcular_stft_y_mel(y, sr)
    inestabilidad = calcular_regularidad_fase(stft_compleja=stft_compleja)

    detector = _get_detector()
    device = detector.device
    modelo = detector.model
    prob_final, predicciones_por_ventana, tiempos_ventanas = predecir_audio_completo(
        mel_db, modelo, device=device
    )

    veredicto = "DEEPFAKE" if prob_final > 50 else "REAL"
    confianza = prob_final if veredicto == "DEEPFAKE" else 100 - prob_final

    pdf_buf = generar_reporte_forense(
        nombre_archivo=os.path.basename(ruta_archivo),
        audio_y=y,
        sr=sr,
        porcentaje_ia=prob_final,
        inestabilidad_fase=inestabilidad,
        tiempos_ventanas=tiempos_ventanas,
        predicciones_por_ventana=predicciones_por_ventana,
        idioma=lang,
    )
    pdf_base64_str = base64.b64encode(pdf_buf.read()).decode("utf-8")

    return AnalyzeResponse(
        veredicto=veredicto,
        confianza=round(confianza, 2),
        inestabilidad=round(inestabilidad, 4),
        pdf_base64=pdf_base64_str,
    )

@app.post("/analyze_file", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    lang: str = Query("es", description="Idioma del reporte: es|en"),
):
    if lang not in ("es", "en"):
        lang = "es"

    if file.content_type not in ("audio/wav", "audio/x-wav", "video/mp4", "audio/mpeg", "audio/mp3", "audio/m4a"):
        ext_ok = (".wav", ".mp4", ".mp3", ".m4a")
        if not any(file.filename.lower().endswith(e) for e in ext_ok):
            raise HTTPException(
                status_code=400,
                detail="Solo se aceptan archivos .wav, .mp4, .mp3 o .m4a",
            )

    id_unico = str(uuid.uuid4())[:8]
    ruta_destino = os.path.join(TEMP_DIR, f"upload_{id_unico}_{file.filename}")
    contents = await file.read()
    with open(ruta_destino, "wb") as f:
        f.write(contents)

    try:
        return _procesar_archivo(ruta_destino, lang=lang)
    finally:
        if os.path.isfile(ruta_destino):
            os.remove(ruta_destino)

@app.post("/analyze_url", response_model=AnalyzeURLResponse)
async def analyze_url(
    body: AnalyzeURLRequest,
    lang: str = Query("es", description="Idioma del reporte: es|en"),
):
    if lang not in ("es", "en"):
        lang = "es"

    try:
        ruta_descargada = descargar_video_de_red_social(body.url, carpeta_destino=TEMP_DIR)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        resp = _procesar_archivo(ruta_descargada, lang=lang)
        return AnalyzeURLResponse(
            **resp.model_dump(),
            archivo_descargado=os.path.basename(ruta_descargada),
        )
    finally:
        if os.path.isfile(ruta_descargada):
            os.remove(ruta_descargada)

@app.get("/health")
async def health():
    return {"status": "ok", "modelo_cargado": _DETECTOR is not None}

if __name__ == "__main__":
    uvicorn.run("servidor_api:app", host="127.0.0.1", port=8000, reload=True)
