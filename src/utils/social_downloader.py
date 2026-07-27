import os
import uuid
import urllib.parse

_DOMINIOS_VALIDOS = frozenset({
    "facebook.com", "www.facebook.com", "fb.watch",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
    "x.com", "www.x.com",
    "twitter.com", "www.twitter.com",
})

def _dominio_es_valido(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        # Extraer dominio base quitando subdominios (p.ej. "m.facebook.com")
        partes = netloc.split(".")
        if len(partes) >= 2:
            base = partes[-2] + "." + partes[-1]
            return base in _DOMINIOS_VALIDOS or netloc in _DOMINIOS_VALIDOS
        return netloc in _DOMINIOS_VALIDOS
    except Exception:
        return False


def descargar_video_de_red_social(url: str, carpeta_destino: str = "data") -> str:
    """
    Descarga un video desde una URL de Facebook, Instagram, TikTok o X/Twitter
    usando yt-dlp, y devuelve la ruta local al archivo .mp4 descargado.

    Lanza ValueError si la URL no corresponde a una plataforma soportada,
    o RuntimeError si la descarga falla.

    LIMITACIÓN CONOCIDA (no resuelta aquí):
    Instagram y Facebook frecuentemente requieren cookies de sesión autenticada
    para contenido que no sea completamente público, y aplican rate-limiting
    agresivo a IPs sin autenticación. TikTok y X suelen funcionar mejor sin
    autenticación para contenido público. Para soporte más robusto de IG/FB,
    la solución es pasar un archivo de cookies exportado por el usuario
    (parámetro cookiefile de yt-dlp).
    """
    if not _dominio_es_valido(url):
        raise ValueError(
            "URL no soportada. Los dominios aceptados son: "
            "facebook.com, fb.watch, instagram.com, tiktok.com, x.com, twitter.com"
        )

    os.makedirs(carpeta_destino, exist_ok=True)

    id_unico = str(uuid.uuid4())[:8]
    ruta_salida = os.path.join(carpeta_destino, f"social_video_{id_unico}.mp4")

    try:
        import yt_dlp
        ydl_opts = {
            "format": "bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": ruta_salida,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 60,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.isfile(ruta_salida):
            raise RuntimeError("No se pudo acceder al contenido. Puede ser privado, haber sido eliminado, o requerir inicio de sesión en la plataforma.")

        return os.path.abspath(ruta_salida)

    except ImportError:
        raise RuntimeError(
            "La librería yt-dlp no está instalada. Ejecute: pip install yt-dlp --upgrade"
        )
    except ValueError:
        raise
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(
            "No se pudo acceder al contenido. Puede ser privado, haber sido eliminado, "
            "o requerir inicio de sesión en la plataforma."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Error inesperado al descargar: {e}") from e
