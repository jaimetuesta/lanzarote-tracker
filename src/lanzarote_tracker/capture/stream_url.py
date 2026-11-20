"""Extraccion de la URL directa del directo de YouTube con yt-dlp.

La camara de Flightradar24 del aeropuerto de Lanzarote se emite por YouTube.
`cv2.VideoCapture` no sabe abrir una pagina de YouTube, pero si el manifiesto
HLS (.m3u8) que hay detras, y es yt-dlp quien lo resuelve.

Este modulo SI esta implementado: es el codigo que ya funcionaba en el primer
`test.py` del repositorio, extraido a una funcion reutilizable.

Fuera de las practicas: yt-dlp no se vio en clase; en el informe se justifica
como herramienta de acceso al video, no como tecnica de vision.
"""

from __future__ import annotations

from typing import Optional

from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import CaptureConfig

_logger = get_logger(__name__)


def get_stream_url(
    youtube_url: Optional[str] = None,
    config: Optional[CaptureConfig] = None,
) -> str:
    """Extrae la URL directa del manifiesto HLS de un directo de YouTube.

    Se piden los clientes `tv`, `ios` y `android` en vez del cliente web porque
    este ultimo devuelve el error "The page needs to be reloaded" y no entrega
    una URL utilizable.

    Args:
        youtube_url: URL de la pagina del directo. Si es None se usa
            `config.stream_url`.
        config: Configuracion de captura, de la que se toman la altura maxima y
            los clientes de yt-dlp.

    Returns:
        La URL directa del stream, lista para `cv2.VideoCapture`.

    Raises:
        RuntimeError: Si yt-dlp no esta instalado o no consigue extraer la URL.
    """
    config = config or CaptureConfig()
    youtube_url = youtube_url or config.stream_url

    try:
        import yt_dlp
    except ImportError as error:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "Falta yt-dlp. Instalalo con `pip install yt-dlp` o crea el entorno "
            "con `conda env create -f environment.yml`."
        ) from error

    options = {
        "format": f"best[height<={config.max_height}]/best",
        "quiet": True,
        "no_warnings": True,
        # Clave para evitar "The page needs to be reloaded": clientes tv / ios.
        "extractor_args": {
            "youtube": {"player_client": list(config.ytdlp_player_clients)}
        },
    }

    _logger.info("Extrayendo la URL del stream con yt-dlp: %s", youtube_url)
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(youtube_url, download=False)
    except Exception as error:  # yt-dlp lanza excepciones propias muy variadas
        raise RuntimeError(f"yt-dlp no pudo extraer el stream: {error}") from error

    url = info.get("url") if isinstance(info, dict) else None
    if not url:
        raise RuntimeError(
            "yt-dlp no devolvio ninguna URL directa. Comprueba que el directo "
            "sigue emitiendo y que yt-dlp esta actualizado."
        )
    return str(url)


__all__ = ["get_stream_url"]
