import os
import random
import shutil


def separar_datos_prueba(proporcion=0.2, semilla=42):
    random.seed(semilla)
    print(f" Iniciando la separación de datos (Train/Test Split) con semilla={semilla}...")
    base_dir = "data"

    for clase in ["real", "fake"]:
        carpeta_train = os.path.join(base_dir, "train", clase)
        carpeta_test = os.path.join(base_dir, "test", clase)

        os.makedirs(carpeta_test, exist_ok=True)

        archivos_train = [f for f in os.listdir(carpeta_train) if f.endswith(".wav")]
        archivos_test_existentes = {f for f in os.listdir(carpeta_test) if f.endswith(".wav")}

        # Solo considerar archivos que aún no están en test
        archivos_disponibles = [f for f in archivos_train if f not in archivos_test_existentes]

        if not archivos_disponibles:
            print(f" No hay archivos nuevos de '{clase.upper()}' para mover a test.")
            continue

        cantidad_test = max(1, int(len(archivos_train) * proporcion))
        cantidad_a_copiar = min(cantidad_test, len(archivos_disponibles))

        archivos_seleccionados = random.sample(archivos_disponibles, cantidad_a_copiar)

        for archivo in archivos_seleccionados:
            origen = os.path.join(carpeta_train, archivo)
            destino = os.path.join(carpeta_test, archivo)
            shutil.copy2(origen, destino)

        print(f" Copiados {cantidad_a_copiar} audios de '{clase.upper()}' a {carpeta_test}")

    print(" Split completado. Train intacto (se copió, no se movió).")


if __name__ == "__main__":
    separar_datos_prueba()