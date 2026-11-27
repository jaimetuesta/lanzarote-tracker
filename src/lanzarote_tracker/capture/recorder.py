"""Grabacion de clips del directo con metadatos.

El sistema base NO trabaja en directo: graba clips del stream y luego trabaja
sobre ellos. Cada clip lleva un JSON con la marca de tiempo UTC de inicio, que
es lo que despues permite cruzarlo con las fichas de vuelo registradas por
:mod:`lanzarote_tracker.identification.live_tracker`.

RECUERDA: hay que grabar con `scripts/track_flights.py` corriendo a la vez. Los
datos ADS-B son efimeros y un clip grabado sin el seguimiento en marcha se
queda sin ficha.

Los clips NO se publican: `data/clips/` esta en .gitignore.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from lanzarote_tracker.capture.stream_url import get_stream_url
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import CaptureConfig, PathsConfig

_logger = get_logger(__name__)


@dataclass(frozen=True)
class ClipMetadata:
    """Metadatos de un clip grabado del directo.

    Attributes:
        clip_path: Ruta del fichero de video.
        started_at_utc: Marca de tiempo UTC del primer fotograma, en ISO 8601.
        started_at_unix: La misma marca en segundos UNIX, que es el formato con
            el que trabajan las fichas de vuelo y OpenSky.
        duration_seconds: Duracion real grabada.
        fps: FPS con los que se escribio el clip, medidos del stream real.
        effective_fps: FPS que resultan de dividir los fotogramas escritos
            entre la duracion. ES EL QUE HAY QUE USAR para situar un fotograma
            en el tiempo y cruzarlo con las fichas de vuelo.
        width: Ancho del fotograma en pixeles.
        height: Alto del fotograma en pixeles.
        frame_count: Numero de fotogramas escritos.
        source_url: URL de la pagina del directo del que procede.
    """

    clip_path: Path
    started_at_utc: str
    started_at_unix: int
    duration_seconds: float
    fps: float
    effective_fps: float
    width: int
    height: int
    frame_count: int
    source_url: str = ""

    def to_json(self) -> dict:
        """Devuelve los metadatos listos para serializar.

        Returns:
            Un diccionario con la ruta convertida a texto.
        """
        payload = asdict(self)
        payload["clip_path"] = str(self.clip_path)
        return payload


def build_clip_name(started_at: datetime, config: CaptureConfig) -> str:
    """Construye el nombre de fichero de un clip a partir de su hora de inicio.

    Args:
        started_at: Hora UTC del primer fotograma.
        config: Configuracion de captura, de la que se toman el formato de la
            marca de tiempo y la extension.

    Returns:
        Un nombre del tipo "clip_20261127T173000Z.mp4".
    """
    stamp = started_at.astimezone(timezone.utc).strftime(config.timestamp_format)
    return f"clip_{stamp}{config.clip_extension}"


def metadata_path_for(clip_path: Path, config: CaptureConfig) -> Path:
    """Devuelve la ruta del JSON de metadatos que acompana a un clip.

    Args:
        clip_path: Ruta del fichero de video.
        config: Configuracion de captura.

    Returns:
        La ruta del JSON, con el mismo nombre y otra extension.
    """
    return clip_path.with_suffix(config.metadata_suffix)


def _now_seconds() -> float:
    """Devuelve un reloj monotono en segundos.

    Returns:
        Segundos desde un origen arbitrario, sin saltos por cambios de hora.
    """
    return cv2.getTickCount() / cv2.getTickFrequency()


def nominal_fps(capture: cv2.VideoCapture, config: CaptureConfig) -> float:
    """Devuelve los FPS que DECLARA el stream, para la cabecera del fichero.

    Es un valor nominal y puede no coincidir con el ritmo al que realmente se
    capturan los fotogramas: un directo de 1080p declara 60 FPS y, segun la red
    y lo que tarde la maquina en decodificar, se capturan menos. La diferencia
    solo afecta a la velocidad de reproduccion del clip.

    Para situar un fotograma EN EL TIEMPO no se usa este valor, sino
    `ClipMetadata.effective_fps`, que se calcula al terminar dividiendo los
    fotogramas escritos entre la duracion medida con el reloj. Ese si es
    correcto por construccion, y es del que depende que el cruce con las fichas
    de vuelo caiga en el sitio.

    NOTA: no sirve medir el ritmo real al principio leyendo unos segundos. Al
    abrir un stream HLS hay un bufer ya descargado que se lee de golpe, y la
    medida sale disparatada (se midieron 153 FPS en un stream de 60).

    Args:
        capture: Captura ya abierta.
        config: Configuracion de captura.

    Returns:
        Los FPS declarados si son plausibles, o el valor de reserva.
    """
    reported = float(capture.get(cv2.CAP_PROP_FPS))
    if 1.0 <= reported <= 120.0:
        return reported
    _logger.debug(
        "El stream informa %.2f FPS, que no es utilizable; se usan %.1f.",
        reported,
        config.fallback_fps,
    )
    return config.fallback_fps


def open_stream(stream_url: str) -> cv2.VideoCapture:
    """Abre el stream del directo con el backend FFMPEG.

    Args:
        stream_url: URL directa del stream.

    Returns:
        La captura ya abierta.

    Raises:
        RuntimeError: Si OpenCV no puede abrirla.
    """
    capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "No se pudo abrir el stream en OpenCV. Comprueba la conexion y que "
            "el directo sigue emitiendo."
        )
    return capture


def record_from_capture(
    capture: cv2.VideoCapture,
    output_dir: Path,
    seconds: float,
    config: CaptureConfig,
    source_url: str = "",
) -> ClipMetadata:
    """Graba un clip desde un stream YA ABIERTO y escribe su JSON.

    Separarlo de :func:`record_clip` permite tener el stream abierto esperando
    y empezar a escribir en cuanto hace falta. Abrirlo tarda unos segundos, y
    esos segundos son justo los del aterrizaje que se quiere capturar.

    Args:
        capture: Captura ya abierta del directo.
        output_dir: Directorio donde se guardan el clip y su JSON.
        seconds: Duracion a grabar.
        config: Configuracion de captura.
        source_url: URL de la pagina del directo, solo informativa.

    Returns:
        Los metadatos del clip grabado.

    Raises:
        RuntimeError: Si no llega ningun fotograma o no se puede crear el
            VideoWriter.
    """
    writer: Optional[cv2.VideoWriter] = None
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                "El stream se abrio pero no entrego ningun fotograma. Vuelve a "
                "intentarlo: suele ser un corte momentaneo del directo."
            )

        height, width = frame.shape[:2]
        fps = nominal_fps(capture, config)

        # A partir de aqui ya se graba: esta es la hora real del clip.
        started_at = datetime.now(timezone.utc)
        started_monotonic = _now_seconds()
        output_dir.mkdir(parents=True, exist_ok=True)
        clip_path = output_dir / build_clip_name(started_at, config)

        fourcc = cv2.VideoWriter_fourcc(*config.fourcc)
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(
                f"No se pudo crear el fichero de video {clip_path} con el codec "
                f"{config.fourcc!r}. Prueba otro codec en config.capture.fourcc."
            )

        _logger.info(
            "Grabando %.0f s en %s (%dx%d a %.1f FPS)...",
            seconds,
            clip_path.name,
            width,
            height,
            fps,
        )

        writer.write(frame)
        frame_count = 1
        elapsed = 0.0
        while elapsed < seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                _logger.warning(
                    "El stream se ha cortado tras %.1f s; se cierra el clip con "
                    "lo grabado.",
                    elapsed,
                )
                break
            writer.write(frame)
            frame_count += 1
            elapsed = _now_seconds() - started_monotonic

        metadata = ClipMetadata(
            clip_path=clip_path,
            started_at_utc=started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            started_at_unix=int(started_at.timestamp()),
            duration_seconds=round(elapsed, 2),
            fps=fps,
            effective_fps=round(frame_count / elapsed, 2) if elapsed > 0 else fps,
            width=width,
            height=height,
            frame_count=frame_count,
            source_url=source_url,
        )
    finally:
        if writer is not None:
            writer.release()

    json_path = metadata_path_for(clip_path, config)
    json_path.write_text(
        json.dumps(metadata.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if metadata.fps > 0 and abs(metadata.effective_fps - metadata.fps) / metadata.fps > 0.2:
        _logger.info(
            "Se han capturado %.1f FPS frente a los %.1f declarados: el clip se "
            "reproducira a otra velocidad, pero las marcas de tiempo usan "
            "effective_fps y siguen siendo correctas.",
            metadata.effective_fps,
            metadata.fps,
        )
    _logger.info(
        "Clip guardado: %s (%d fotogramas, %.1f s a %.1f FPS, inicio %s)",
        clip_path.name,
        frame_count,
        metadata.duration_seconds,
        metadata.effective_fps,
        metadata.started_at_utc,
    )
    return metadata


def record_clip(
    stream_url: str,
    output_dir: Path,
    seconds: float,
    config: CaptureConfig,
    source_url: str = "",
) -> ClipMetadata:
    """Abre el stream, graba un clip y lo cierra.

    La marca de tiempo se toma JUSTO antes de escribir el primer fotograma, no
    al abrir el stream: abrirlo puede tardar varios segundos y ese desfase
    estropearia el cruce con las fichas de vuelo.

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
        RuntimeError: Si no se puede abrir el stream o grabar.
    """
    capture = open_stream(stream_url)
    try:
        return record_from_capture(capture, output_dir, seconds, config, source_url)
    finally:
        capture.release()


def load_clip_metadata(clip_path: Path, config: CaptureConfig) -> ClipMetadata:
    """Lee el JSON de metadatos que acompana a un clip.

    Args:
        clip_path: Ruta del fichero de video.
        config: Configuracion de captura, de la que se toma el sufijo del JSON.

    Returns:
        Los metadatos del clip.

    Raises:
        FileNotFoundError: Si no existe el JSON junto al clip. Sin el no se
            sabe cuando se grabo y el clip no se puede cruzar con las fichas.
    """
    json_path = metadata_path_for(Path(clip_path), config)
    if not json_path.is_file():
        raise FileNotFoundError(
            f"No existe {json_path}. Sin la marca de tiempo no se puede saber "
            "que vuelo sale en el clip."
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["clip_path"] = Path(payload["clip_path"])
    return ClipMetadata(**payload)


def clip_time_utc(
    metadata: ClipMetadata,
    offset_seconds: float = 0.0,
) -> datetime:
    """Devuelve la hora UTC de un instante concreto dentro del clip.

    Es lo que hace falta para cruzar un FOTOGRAMA con las fichas de vuelo: no
    vale la hora de inicio del clip si el avion aparece en el segundo 40.

    Args:
        metadata: Metadatos del clip.
        offset_seconds: Segundos transcurridos desde el inicio del clip.

    Returns:
        La hora UTC de ese instante.
    """
    from datetime import timedelta

    started = datetime.fromtimestamp(metadata.started_at_unix, tz=timezone.utc)
    return started + timedelta(seconds=offset_seconds)


def record_clips(
    count: int,
    paths: PathsConfig,
    config: CaptureConfig,
    seconds: Optional[float] = None,
) -> Tuple[ClipMetadata, ...]:
    """Graba varios clips seguidos del directo.

    La URL del stream se resuelve UNA vez y se reutiliza: extraerla con yt-dlp
    tarda varios segundos y no hace falta repetirlo en cada clip.

    Args:
        count: Numero de clips a grabar.
        paths: Rutas del proyecto, de las que se toma `clips`.
        config: Configuracion de captura.
        seconds: Duracion de cada clip. Si es None se usa `config.clip_seconds`.

    Returns:
        Los metadatos de todos los clips grabados correctamente.

    Raises:
        RuntimeError: Si no se puede obtener la URL del stream.
    """
    seconds = config.clip_seconds if seconds is None else seconds
    stream_url = get_stream_url(config.stream_url, config)

    recorded: List[ClipMetadata] = []
    for index in range(count):
        try:
            recorded.append(
                record_clip(
                    stream_url,
                    Path(paths.clips),
                    seconds,
                    config,
                    source_url=config.stream_url,
                )
            )
        except RuntimeError as error:
            _logger.error("Fallo al grabar el clip %d de %d: %s", index + 1, count, error)
            # La URL de un directo caduca; se renueva y se sigue.
            try:
                stream_url = get_stream_url(config.stream_url, config)
                _logger.info("URL del stream renovada; se continua.")
            except RuntimeError as refresh_error:
                _logger.error("Tampoco se pudo renovar la URL: %s", refresh_error)
                break
    return tuple(recorded)


__all__ = [
    "ClipMetadata",
    "nominal_fps",
    "open_stream",
    "record_from_capture",
    "build_clip_name",
    "metadata_path_for",
    "record_clip",
    "load_clip_metadata",
    "clip_time_utc",
    "record_clips",
]
