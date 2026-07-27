import os
import sys
import random
import shutil
import subprocess
import kagglehub
import imageio_ffmpeg

RANDOM_SEED = 42
TEST_SPLIT = 0.2
TARGET_SR = 16000
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")

random.seed(RANDOM_SEED)


def descargar_fakeavceleb():
    print("Descargando FakeAVCeleb desde Kaggle...")
    ruta = kagglehub.dataset_download("shreyaty08/fakeavceleb")
    print(f"Dataset descargado en: {ruta}")
    return ruta


def extraer_audio(video_path, output_path):
    cmd = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", str(TARGET_SR),
        "-ac", "1",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(output_path)


def es_real(categoria):
    return categoria == "RealVideo-RealAudio"


CATEGORIAS_VALIDAS = ("FakeVideo-FakeAudio",
                     "RealVideo-FakeAudio", "RealVideo-RealAudio")

def recopilar_videos(dataset_root):
    cats_dir = None
    for root, dirs, files in os.walk(dataset_root):
        found = [d for d in dirs if d in CATEGORIAS_VALIDAS]
        if found:
            cats_dir = root
            break

    if cats_dir is None:
        print("ERROR: no se encontro el directorio de categorias")
        return []

    videos = []
    print(f"Buscando videos en: {cats_dir}")
    for cat in sorted(os.listdir(cats_dir)):
        cat_path = os.path.join(cats_dir, cat)
        if not os.path.isdir(cat_path) or cat not in CATEGORIAS_VALIDAS:
            continue
        print(f"  Categoria: {cat}")
        for etnia in sorted(os.listdir(cat_path)):
            etnia_path = os.path.join(cat_path, etnia)
            if not os.path.isdir(etnia_path):
                continue
            for genero in sorted(os.listdir(etnia_path)):
                genero_path = os.path.join(etnia_path, genero)
                if not os.path.isdir(genero_path):
                    continue
                for ident in sorted(os.listdir(genero_path)):
                    ident_path = os.path.join(genero_path, ident)
                    if not os.path.isdir(ident_path):
                        continue
                    for archivo in sorted(os.listdir(ident_path)):
                        if archivo.endswith(".mp4"):
                            videos.append({
                                "ruta": os.path.join(ident_path, archivo),
                                "archivo": archivo,
                                "categoria": cat,
                                "etiqueta": 0 if es_real(cat) else 1,
                                "nombre_base": os.path.splitext(archivo)[0]
                            })
    return videos


def preparar_directorios():
    base = "./data"
    for split in ["train", "test"]:
        for clase in ["real", "fake"]:
            os.makedirs(os.path.join(base, split, clase), exist_ok=True)


def limpiar_datos_viejos():
    base = "./data"
    for split in ["train", "test"]:
        for clase in ["real", "fake"]:
            dir_path = os.path.join(base, split, clase)
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    fpath = os.path.join(dir_path, f)
                    if os.path.isfile(fpath) and (f.endswith(".wav") or f.endswith(".mp4")):
                        try:
                            os.remove(fpath)
                        except PermissionError:
                            print(f"  No se pudo eliminar (en uso): {fpath}")


def main():
    dataset_root = descargar_fakeavceleb()
    print(f"Explorando directorio: {dataset_root}")
    for item in os.listdir(dataset_root):
        print(f"  - {item}")

    videos = recopilar_videos(dataset_root)

    if not videos:
        print("ERROR: No se encontraron videos .mp4 en el dataset.")
        sys.exit(1)

    print(f"Total videos encontrados: {len(videos)}")
    real_count = sum(1 for v in videos if v["etiqueta"] == 0)
    fake_count = sum(1 for v in videos if v["etiqueta"] == 1)
    print(f"  Reales: {real_count}, Fakes: {fake_count}")

    random.shuffle(videos)
    n_test = max(1, int(len(videos) * TEST_SPLIT))
    test_videos = videos[:n_test]
    train_videos = videos[n_test:]

    preparar_directorios()

    for split_name, split_videos in [("train", train_videos), ("test", test_videos)]:
        print(f"Procesando {split_name} ({len(split_videos)} videos)...")
        for i, v in enumerate(split_videos):
            clase = "real" if v["etiqueta"] == 0 else "fake"
            nombre_wav = f"{v['categoria']}_{v['nombre_base']}.wav"
            nombre_wav = nombre_wav.replace("/", "_").replace("\\", "_")
            output_path = os.path.join("./data", split_name, clase, nombre_wav)
            if os.path.exists(output_path):
                continue
            extraer_audio(v["ruta"], output_path)
            if (i + 1) % 50 == 0:
                print(f"  {split_name}: {i+1}/{len(split_videos)}")

    print("\nVerificando resultado final...")
    real_train = len([f for f in os.listdir("./data/train/real") if f.endswith(".wav")]) if os.path.exists("./data/train/real") else 0
    fake_train = len([f for f in os.listdir("./data/train/fake") if f.endswith(".wav")]) if os.path.exists("./data/train/fake") else 0
    real_test = len([f for f in os.listdir("./data/test/real") if f.endswith(".wav")]) if os.path.exists("./data/test/real") else 0
    fake_test = len([f for f in os.listdir("./data/test/fake") if f.endswith(".wav")]) if os.path.exists("./data/test/fake") else 0
    print(f"Train: {real_train} real, {fake_train} fake")
    print(f"Test:  {real_test} real, {fake_test} fake")
    print("Listo!")


if __name__ == "__main__":
    main()
