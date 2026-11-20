"""Registro de imagenes intermedias: la secuencia de transformacion del informe.

El enunciado exige una figura con TODAS las transformaciones que sufre la
imagen. Cada pipeline del proyecto (segmentacion de tarjetas, deteccion del
avion, etc.) recibe un :class:`DebugRecorder` opcional y le va entregando sus
pasos intermedios con un nombre. Si la depuracion esta desactivada el recorder
no hace nada, asi que la llamada puede quedarse en el codigo de produccion sin
coste apreciable.

Uso tipico::

    recorder = DebugRecorder("security", config.debug)
    recorder.record("original", frame)
    recorder.record("hsv", hsv)
    recorder.record("mask", mask)
    recorder.save_all()   # -> results/security/00_original.png, 01_hsv.png, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from lanzarote_tracker.common.io_utils import write_image
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import DebugConfig

Image = np.ndarray

_logger = get_logger(__name__)


def to_bgr(image: Image) -> Image:
    """Convierte una imagen a BGR de 3 canales para poder mostrarla o apilarla.

    Acepta mascaras binarias, imagenes en escala de grises e imagenes BGR.

    Args:
        image: Imagen de 2 o 3 canales, de cualquier dtype.

    Returns:
        Una imagen BGR de 3 canales y dtype uint8.
    """
    array = image
    if array.dtype != np.uint8:
        # Reescala a 0-255 para poder guardarla; las mascaras de 0/1 se veran bien.
        maximum = float(array.max()) if array.size else 0.0
        if maximum <= 1.0:
            array = (array * 255.0).astype(np.uint8)
        else:
            array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    return array


class DebugRecorder:
    """Acumula los pasos intermedios de un pipeline y los guarda en disco.

    Attributes:
        module_name: Nombre del modulo, que da el subdirectorio de salida.
        config: Configuracion de depuracion.
        steps: Pasos registrados, como pares (nombre, imagen).
    """

    def __init__(self, module_name: str, config: Optional[DebugConfig] = None) -> None:
        """Inicializa el recorder.

        Args:
            module_name: Nombre del modulo, p. ej. "security" o "detection".
                Se usa como subdirectorio dentro de `config.output_root`.
            config: Configuracion de depuracion. Si es None se usa la de por
                defecto, que tiene la depuracion DESACTIVADA.
        """
        self.module_name: str = module_name
        self.config: DebugConfig = config or DebugConfig()
        self.steps: List[Tuple[str, Image]] = []

    @property
    def enabled(self) -> bool:
        """Indica si el registro esta activado."""
        return self.config.enabled

    @property
    def output_dir(self) -> Path:
        """Directorio donde se guardan los pasos de este modulo."""
        return Path(self.config.output_root) / self.module_name

    def record(self, name: str, image: Image) -> None:
        """Registra un paso intermedio del pipeline.

        No hace nada si la depuracion esta desactivada. La imagen se COPIA, para
        que una modificacion posterior del pipeline no altere lo registrado.

        Args:
            name: Nombre descriptivo del paso, p. ej. "hsv_mask_red".
            image: Imagen del paso.
        """
        if not self.enabled:
            return
        self.steps.append((name, image.copy()))

    def save_all(self, prefix: str = "") -> List[Path]:
        """Guarda en disco todos los pasos registrados, numerados en orden.

        Args:
            prefix: Prefijo opcional para los nombres de fichero, util para
                distinguir varios fotogramas dentro de la misma ejecucion.

        Returns:
            Lista de rutas escritas. Vacia si la depuracion esta desactivada.
        """
        if not self.enabled or not self.steps:
            return []
        written: List[Path] = []
        for index, (name, image) in enumerate(self.steps):
            filename = f"{prefix}{index:02d}_{name}{self.config.image_extension}"
            written.append(write_image(self.output_dir / filename, to_bgr(image)))
        if self.config.save_mosaic:
            mosaic = self.build_mosaic()
            if mosaic is not None:
                mosaic_name = f"{prefix}mosaic{self.config.image_extension}"
                written.append(write_image(self.output_dir / mosaic_name, mosaic))
        _logger.debug("Guardados %d pasos en %s", len(written), self.output_dir)
        return written

    def build_mosaic(self, titles: bool = True) -> Optional[Image]:
        """Compone todos los pasos en una sola figura para el informe.

        Args:
            titles: Si se escribe el nombre de cada paso sobre su casilla.

        Returns:
            El mosaico como imagen BGR, o None si no hay pasos registrados.
        """
        if not self.steps:
            return None
        tiles = [
            _make_tile(image, name if titles else "", self.config.mosaic_tile_height)
            for name, image in self.steps
        ]
        return _grid(tiles, self.config.mosaic_columns)

    def clear(self) -> None:
        """Olvida los pasos registrados hasta ahora."""
        self.steps.clear()


def _make_tile(image: Image, title: str, height: int) -> Image:
    """Reescala una imagen a una altura fija y le escribe su titulo encima.

    Args:
        image: Imagen del paso.
        title: Texto a escribir; cadena vacia para no escribir nada.
        height: Altura destino en pixeles.

    Returns:
        La casilla del mosaico, en BGR.
    """
    tile = to_bgr(image)
    scale = height / float(tile.shape[0])
    width = max(int(round(tile.shape[1] * scale)), 1)
    tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)
    if title:
        cv2.rectangle(tile, (0, 0), (width, 22), (0, 0, 0), thickness=-1)
        cv2.putText(
            tile, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
    return tile


def _grid(tiles: Sequence[Image], columns: int) -> Image:
    """Apila casillas de la misma altura en una rejilla.

    Args:
        tiles: Casillas, todas con la misma altura.
        columns: Numero de columnas de la rejilla.

    Returns:
        La rejilla completa, rellenada con negro donde falten casillas.
    """
    columns = max(columns, 1)
    height = tiles[0].shape[0]
    rows: List[Image] = []
    for start in range(0, len(tiles), columns):
        chunk = list(tiles[start : start + columns])
        row = np.hstack(chunk)
        rows.append(row)
    max_width = max(row.shape[1] for row in rows)
    padded = [
        np.hstack([row, np.zeros((height, max_width - row.shape[1], 3), dtype=np.uint8)])
        if row.shape[1] < max_width
        else row
        for row in rows
    ]
    return np.vstack(padded)


__all__ = ["DebugRecorder", "to_bgr"]
