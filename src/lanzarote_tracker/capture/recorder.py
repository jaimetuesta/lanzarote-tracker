"""Grabacion de clips del directo con metadatos.

El sistema base NO trabaja en directo: graba clips del stream y luego trabaja
sobre ellos. Cada clip lleva un JSON con la marca de tiempo UTC de inicio, que
es lo que despues permite cruzarlo con las consultas de OpenSky.

Los clips NO se publican: `data/clips/` esta en .gitignore.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from lanzarote_tracker.config import CaptureConfig, PathsConfig


@dataclass(frozen=True)
class ClipMetadata:
    """Metadatos de un clip grabado del directo.

    Attributes:
        clip_path: Ruta del fichero de video.
        started_at_utc: Marca de tiempo UTC del primer fotograma, en ISO 8601.
        duration_seconds: Duracion real grabada.
        fps: FPS con los que se escribio el clip.
        width: Ancho del fotograma en pixeles.
        height: Alto del fotograma en pixeles.
        frame_count: Numero de fotogramas escritos.
        source_url: URL de la pagina del directo del que procede.
    """

    clip_path: Path
    started_at_utc: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    source_url: str


def build_clip_name(started_at: datetime, config: CaptureConfig) -> str:
    """Construye el nombre de fichero de un clip a partir de su hora de inicio.

    Args:
        started_at: Hora UTC del primer fotograma.
        config: Configuracion de captura, de la que se toman el formato de la
            marca de tiempo y la extension.

    Returns:
        Un nombre del tipo "clip_20261120T173000Z.mp4".
    """
    raise NotImplementedError(
        "Pendiente: build_clip_name (capture/recorder.py). "
        "started_at.strftime(config.timestamp_format) + config.clip_extension."
    )


def record_clip(
    stream_url: str,
    output_dir: Path,
    seconds: float,
    config: CaptureConfig,
    source_url: str = "",
) -> ClipMetadata:
    """Graba un clip del stream y escribe su JSON de metadatos.

    Abre el stream con `cv2.VideoCapture`, escribe los fotogramas con
    `cv2.VideoWriter` usando el FourCC de la configuracion, y al terminar
    guarda junto al video un JSON con la marca de tiempo UTC de inicio.

    La marca de tiempo se toma JUSTO antes de escribir el primer fotograma, no
    al abrir el stream, porque abrirlo puede tardar varios segundos y el desfase
    estropearia el cruce con OpenSky.

    Args:
        stream_url: URL directa del stream, la que devuelve
            `capture.stream_url.get_stream_url`.
        output_dir: Directorio donde se guardan el clip y su JSON.
        seconds: Duracion a grabar.
        config: Configuracion de captura.
        source_url: URL de la pagina del directo, solo informativa.

    Returns:
        Los metadatos del clip grabado.

    Raises:
        RuntimeError: Si no se puede abrir el stream o crear el VideoWriter.
    """
    raise NotImplementedError(
        "Pendiente: record_clip (capture/recorder.py). "
        "cv2.VideoCapture + cv2.VideoWriter + json.dump de los metadatos."
    )


def load_clip_metadata(clip_path: Path, config: CaptureConfig) -> ClipMetadata:
    """Lee el JSON de metadatos que acompana a un clip.

    Args:
        clip_path: Ruta del fichero de video.
        config: Configuracion de captura, de la que se toma el sufijo del JSON.

    Returns:
        Los metadatos del clip.

    Raises:
        FileNotFoundError: Si no existe el JSON junto al clip.
    """
    raise NotImplementedError(
        "Pendiente: load_clip_metadata (capture/recorder.py). "
        "json.load del fichero con el sufijo config.metadata_suffix."
    )


def record_clips(
    count: int,
    paths: PathsConfig,
    config: CaptureConfig,
    seconds: Optional[float] = None,
) -> Tuple[ClipMetadata, ...]:
    """Graba varios clips seguidos del directo.

    Es el punto de entrada que usa `scripts/record_clips.py`.

    Args:
        count: Numero de clips a grabar.
        paths: Rutas del proyecto, de las que se toma `clips`.
        config: Configuracion de captura.
        seconds: Duracion de cada clip. Si es None se usa `config.clip_seconds`.

    Returns:
        Los metadatos de todos los clips grabados.
    """
    raise NotImplementedError(
        "Pendiente: record_clips (capture/recorder.py). "
        "get_stream_url + bucle de record_clip."
    )


__all__ = [
    "ClipMetadata",
    "build_clip_name",
    "record_clip",
    "load_clip_metadata",
    "record_clips",
]
