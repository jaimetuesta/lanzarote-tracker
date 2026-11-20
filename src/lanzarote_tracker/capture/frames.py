"""Extraccion de fotogramas de los clips grabados y division del dataset.

La division train/val/test se hace SIEMPRE por clips completos, nunca por
fotogramas: dos fotogramas consecutivos del mismo clip son casi identicos, asi
que repartirlos entre particiones dejaria al clasificador ver en entrenamiento
imagenes practicamente iguales a las de validacion, y las metricas saldrian
infladas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from lanzarote_tracker.config import CaptureConfig, ClassificationConfig, PathsConfig


def extract_frames(
    clip_path: Path,
    output_dir: Path,
    config: CaptureConfig,
) -> List[Path]:
    """Extrae fotogramas de un clip, uno de cada `config.frame_step`.

    Los fotogramas se guardan en un subdirectorio con el nombre del clip, para
    no perder de que clip viene cada uno: esa informacion es imprescindible
    luego para dividir el dataset por clips.

    Args:
        clip_path: Ruta del clip grabado.
        output_dir: Directorio raiz donde se crean los subdirectorios.
        config: Configuracion de captura, de la que se toman `frame_step` y
            `frame_extension`.

    Returns:
        Lista de rutas de los fotogramas escritos.

    Raises:
        FileNotFoundError: Si el clip no existe.
    """
    raise NotImplementedError(
        "Pendiente: extract_frames (capture/frames.py). "
        "open_video + bucle de cap.read() guardando uno de cada frame_step."
    )


def extract_all_frames(
    paths: PathsConfig,
    config: CaptureConfig,
) -> Dict[str, List[Path]]:
    """Extrae fotogramas de todos los clips de `paths.clips`.

    Args:
        paths: Rutas del proyecto.
        config: Configuracion de captura.

    Returns:
        Un diccionario del nombre del clip a la lista de fotogramas extraidos.
    """
    raise NotImplementedError(
        "Pendiente: extract_all_frames (capture/frames.py). "
        "Bucle de extract_frames sobre los clips de paths.clips."
    )


def split_by_clip(
    clip_names: Sequence[str],
    config: ClassificationConfig,
) -> Dict[str, Tuple[str, ...]]:
    """Divide los clips en train, val y test segun las proporciones configuradas.

    La division es POR CLIPS COMPLETOS: un clip entero cae en una sola
    particion. El reparto es reproducible porque usa `config.random_state`.

    Args:
        clip_names: Nombres de los clips disponibles.
        config: Configuracion de clasificacion, de la que se toman
            `split_ratios` y `random_state`.

    Returns:
        Un diccionario con las claves "train", "val" y "test" y la tupla de
        nombres de clip de cada particion.

    Raises:
        ValueError: Si las proporciones no suman 1 o no hay clips suficientes.
    """
    raise NotImplementedError(
        "Pendiente: split_by_clip (capture/frames.py). "
        "Barajar los nombres de clip con random.Random(config.random_state) y "
        "repartirlos segun config.split_ratios."
    )


def write_splits(
    splits: Dict[str, Tuple[str, ...]],
    paths: PathsConfig,
) -> List[Path]:
    """Guarda la division del dataset en `data/splits/`.

    Args:
        splits: Division devuelta por :func:`split_by_clip`.
        paths: Rutas del proyecto.

    Returns:
        Las rutas de los ficheros escritos, uno por particion.
    """
    raise NotImplementedError(
        "Pendiente: write_splits (capture/frames.py). "
        "Un fichero de texto por particion con un nombre de clip por linea."
    )


__all__ = ["extract_frames", "extract_all_frames", "split_by_clip", "write_splits"]
