import os
import shutil
import warnings
warnings.filterwarnings('ignore')
import torchaudio

DIRS = ["./data/train", "./data/test"]
CORRUPT_DIR = "./data/corruptos"
CACHE_DIR = "./data/cache"

def sanitizar():
    contador_corruptos = 0
    contador_limpios = 0

    for base_dir in DIRS:
        if not os.path.exists(base_dir):
            continue
        for clase in ["real", "fake"]:
            dir_path = os.path.join(base_dir, clase)
            if not os.path.exists(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path)):
                if not fname.endswith(".wav"):
                    continue
                fpath = os.path.join(dir_path, fname)
                try:
                    wave, sr = torchaudio.load(fpath)
                    contador_limpios += 1
                except Exception as e:
                    print(f"CORRUPTO: {fpath} -> {e}")
                    os.makedirs(CORRUPT_DIR, exist_ok=True)
                    split_name = os.path.basename(base_dir)
                    dest = os.path.join(CORRUPT_DIR, f"{split_name}_{clase}_{fname}")
                    try:
                        os.replace(fpath, dest)
                    except Exception:
                        shutil.copy2(fpath, dest)
                        os.remove(fpath)
                    contador_corruptos += 1

    print(f"\nResumen: {contador_limpios} limpios, {contador_corruptos} corruptos movidos a {CORRUPT_DIR}/")

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        print(f"Cache eliminado: {CACHE_DIR}")

if __name__ == "__main__":
    sanitizar()
