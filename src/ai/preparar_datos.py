"""
Script temporal de preparación de datos.
Ejecutar UNA SOLA VEZ para descargar los datasets de Kaggle (fake y real),
submuestrear 5,000 audios de cada uno de forma aleatoria y copiarlos al
pipeline forense.

Uso:
    python preparar_datos.py
"""

import os
import random
import shutil
from pathlib import Path

import kagglehub

# ------------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------------
NUM_MUESTRAS = 5000


def seleccionar_y_copiar(carpeta_origen, carpeta_destino, num_muestras, extension=".wav"):
    """
    Busca recursivamente en `carpeta_origen` (incluyendo todas sus subcarpetas),
    selecciona aleatoriamente `num_muestras` archivos y los copia a `carpeta_destino`.
    """
    print(f" Buscando archivos recursivamente en: {carpeta_origen}")
    ruta_origen = Path(carpeta_origen)
    todos_los_archivos = list(ruta_origen.rglob(f"*{extension}"))
    print(f" Total de archivos disponibles: {len(todos_los_archivos)}")

    muestras_reales = min(num_muestras, len(todos_los_archivos))

    if muestras_reales < num_muestras:
        print(f" [AVISO] Solo se encontraron {muestras_reales} archivos de los {num_muestras} solicitados.")
        print(" Ajustando el limite automaticamente...")

    archivos_seleccionados = random.sample(todos_los_archivos, muestras_reales)
    print(f" Seleccion aleatoria completada: {len(archivos_seleccionados)} archivos elegidos.")

    os.makedirs(carpeta_destino, exist_ok=True)

    copiados = 0
    errores = 0

    for ruta_archivo in archivos_seleccionados:
        # ruta_archivo ya es la ruta completa y absoluta (objeto Path) lista para copiar
        ruta_destino = os.path.join(carpeta_destino, ruta_archivo.name)

        try:
            shutil.copy(ruta_archivo, ruta_destino)
            copiados += 1
        except Exception as e:
            errores += 1
            print(f" Error copiando {ruta_archivo.name}: {e}")

    print(f" Archivos copiados exitosamente: {copiados}/{len(archivos_seleccionados)}")
    if errores > 0:
        print(f" Archivos con error: {errores}")
    print(f" Destino final: {carpeta_destino}")

    return copiados, errores


# ==================================================================
# 1. DATASET FAKE
# ==================================================================
print(" Descargando dataset FAKE desde Kaggle (esto puede tardar unos minutos)...")

DATASET_KAGGLE_ID_FAKE = "walimuhammadahmad/fakeaudio"

ruta_kaggle_fake = kagglehub.dataset_download(DATASET_KAGGLE_ID_FAKE)
print(f" Dataset FAKE descargado en: {ruta_kaggle_fake}")

# Con rglob ya no es crítico apuntar a la subcarpeta exacta,
# pero se mantiene por si quieres acotar la búsqueda a "fake" específicamente.
CARPETA_ORIGEN_FAKE = ruta_kaggle_fake
CARPETA_DESTINO_FAKE = os.path.join("data", "train", "fake")

copiados_fake, errores_fake = seleccionar_y_copiar(
    CARPETA_ORIGEN_FAKE, CARPETA_DESTINO_FAKE, NUM_MUESTRAS
)

# ==================================================================
# 2. DATASET REAL (voces reales - Speaker Recognition Audio Dataset)
# ==================================================================
print("\n Descargando dataset REAL desde Kaggle (esto puede tardar unos minutos)...")

ruta_kaggle_real = kagglehub.dataset_download(
    "vjcalling/speaker-recognition-audio-dataset"
)
print(f" Dataset REAL descargado en: {ruta_kaggle_real}")

# La búsqueda recursiva con rglob ahora encuentra los .wav sin importar
# en qué subcarpeta de hablantes estén organizados.
CARPETA_ORIGEN_REAL = ruta_kaggle_real
CARPETA_DESTINO_REAL = os.path.join("data", "train", "real")

copiados_real, errores_real = seleccionar_y_copiar(
    CARPETA_ORIGEN_REAL, CARPETA_DESTINO_REAL, NUM_MUESTRAS
)

# ------------------------------------------------------------------
# Resumen final
# ------------------------------------------------------------------
print("\n --- RESUMEN DE PREPARACIÓN DE DATOS ---")
print(f" FAKE -> copiados: {copiados_fake}/{NUM_MUESTRAS} | errores: {errores_fake}")
print(f" REAL -> copiados: {copiados_real}/{NUM_MUESTRAS} | errores: {errores_real}")
print(" Listo. custom_dataset.py ya puede leer ambas carpetas como si siempre hubieran estado ahí.")