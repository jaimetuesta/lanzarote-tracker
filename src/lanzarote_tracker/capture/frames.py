"""Extraccion de fotogramas de los clips grabados y division del dataset.

La division train/val/test se hace SIEMPRE por clips completos, nunca por
fotogramas: dos fotogramas consecutivos del mismo clip son casi identicos, asi
que repartirlos entre particiones dejaria al clasificador ver en entrenamiento
imagenes practicamente iguales a las de validacion, y las metricas saldrian
infladas.

Por eso los fotogramas de cada clip van a su propio subdirectorio: asi nunca se
pierde de que clip viene cada uno.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

from lanzarote_tracker.common.io_utils import open_video, write_image
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import CaptureConfig, ClassificationConfig, PathsConfig

_logger = get_logger(__name__)

#: Nombres de las particiones del dataset, en orden.
SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")


def extract_frames(
    clip_path: Path,
    output_dir: Path,
    config: CaptureConfig,
) -> List[Path]:
    """Extrae fotogramas de un clip, uno de cada `config.frame_step`.

    Los fotogramas se guardan en un subdirectorio con el nombre del clip, para
    no perder de que clip viene cada uno: esa informacion es imprescindible
    luego para dividir el dataset por clips.

    El nombre de cada fotograma lleva su indice dentro del clip, que es lo que
    permite reconstruir en que segundo aparece y cruzarlo con las fichas de
    vuelo.

    Args:
        clip_path: Ruta del clip grabado.
        output_dir: Directorio raiz donde se crean los subdirectorios.
        config: Configuracion de captura, de la que se toman `frame_step` y
            `frame_extension`.

    Returns:
        Lista de rutas de los fotogramas escritos.

    Raises:
        FileNotFoundError: Si el clip no existe.
        ValueError: Si `frame_step` no es positivo.
    """
    if config.frame_step <= 0:
        raise ValueError("config.capture.frame_step debe ser mayor que 0.")

    clip_path = Path(clip_path)
    capture = open_video(clip_path)
    destination = Path(output_dir) / clip_path.stem
    destination.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % config.frame_step == 0:
                name = f"frame_{index:06d}{config.frame_extension}"
                written.append(write_image(destination / name, frame))
            index += 1
    finally:
        capture.release()

    _logger.info(
        "%s: %d fotogramas extraidos de %d leidos.", clip_path.name, len(written), index
    )
    return written


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

    Raises:
        FileNotFoundError: Si el directorio de clips no existe.
    """
    clips_dir = Path(paths.clips)
    if not clips_dir.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de clips {clips_dir}. Graba alguno con "
            "`python scripts/record_clips.py`."
        )

    results: Dict[str, List[Path]] = {}
    for clip_path in sorted(clips_dir.glob(f"*{config.clip_extension}")):
        results[clip_path.stem] = extract_frames(clip_path, Path(paths.frames), config)
    if not results:
        _logger.warning("No hay clips en %s.", clips_dir)
    return results


def split_by_clip(
    clip_names: Sequence[str],
    config: ClassificationConfig,
) -> Dict[str, Tuple[str, ...]]:
    """Divide los clips en train, val y test segun las proporciones configuradas.

    La division es POR CLIPS COMPLETOS: un clip entero cae en una sola
    particion. El reparto es reproducible porque usa `config.random_state`.

    Se garantiza que val y test reciben al menos un clip cuando hay clips
    suficientes: con pocos clips, el redondeo de las proporciones tiende a
    dejarlos vacios y entonces no se puede medir nada.

    Args:
        clip_names: Nombres de los clips disponibles.
        config: Configuracion de clasificacion, de la que se toman
            `split_ratios` y `random_state`.

    Returns:
        Un diccionario con las claves "train", "val" y "test" y la tupla de
        nombres de clip de cada particion.

    Raises:
        ValueError: Si las proporciones no suman 1, si alguna es negativa o si
            hay menos clips que particiones.
    """
    ratios = tuple(config.split_ratios)
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(
            f"Hacen falta {len(SPLIT_NAMES)} proporciones, una por particion."
        )
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Las proporciones no pueden ser negativas.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Las proporciones deben sumar 1 y suman {sum(ratios)}.")

    names = sorted(set(clip_names))
    if len(names) < len(SPLIT_NAMES):
        raise ValueError(
            f"Hacen falta al menos {len(SPLIT_NAMES)} clips para dividir el "
            f"dataset y hay {len(names)}. Graba mas con "
            "`python scripts/record_clips.py`."
        )

    random.Random(config.random_state).shuffle(names)

    # Una plaza garantizada para val y test, y el resto por proporciones.
    remaining = list(names)
    splits: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}
    for name in ("val", "test"):
        splits[name].append(remaining.pop())

    total = len(remaining)
    train_count = int(round(ratios[0] * len(names)))
    train_count = max(1, min(train_count, total))
    splits["train"] = remaining[:train_count]

    rest = remaining[train_count:]
    val_share = ratios[1] / (ratios[1] + ratios[2]) if (ratios[1] + ratios[2]) else 0.5
    val_count = int(round(val_share * len(rest)))
    splits["val"].extend(rest[:val_count])
    splits["test"].extend(rest[val_count:])

    return {name: tuple(sorted(splits[name])) for name in SPLIT_NAMES}


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
    directory = Path(paths.splits)
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name, clip_names in splits.items():
        path = directory / f"{name}.txt"
        path.write_text("\n".join(clip_names) + "\n", encoding="utf-8")
        written.append(path)
        _logger.info("%s: %d clips", name, len(clip_names))
    return written


def read_splits(paths: PathsConfig) -> Dict[str, Tuple[str, ...]]:
    """Lee la division del dataset guardada previamente.

    Args:
        paths: Rutas del proyecto.

    Returns:
        La division, con las particiones que existan en disco.

    Raises:
        FileNotFoundError: Si no hay ninguna particion escrita.
    """
    directory = Path(paths.splits)
    splits: Dict[str, Tuple[str, ...]] = {}
    for name in SPLIT_NAMES:
        path = directory / f"{name}.txt"
        if path.is_file():
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            splits[name] = tuple(line for line in lines if line)
    if not splits:
        raise FileNotFoundError(
            f"No hay particiones en {directory}. Creala con "
            "`python scripts/build_dataset.py`."
        )
    return splits


def frame_offset_seconds(frame_name: str, fps: float, frame_step: int) -> float:
    """Calcula en que segundo del clip aparece un fotograma extraido.

    Hace falta para cruzar un fotograma con las fichas de vuelo: la marca de
    tiempo del clip es la de su inicio, no la del fotograma.

    Args:
        frame_name: Nombre del fichero, del tipo "frame_000120.png".
        fps: FPS del clip, de sus metadatos.
        frame_step: Paso con el que se extrajeron los fotogramas. Se acepta
            para dejar claro que el indice del nombre es el del CLIP, no el del
            conjunto extraido.

    Returns:
        Los segundos transcurridos desde el inicio del clip.

    Raises:
        ValueError: Si el nombre no lleva indice o los FPS no son positivos.
    """
    if fps <= 0:
        raise ValueError("Los FPS del clip deben ser positivos.")
    stem = Path(frame_name).stem
    _, _, digits = stem.partition("_")
    if not digits.isdigit():
        raise ValueError(
            f"El nombre {frame_name!r} no tiene el formato 'frame_<indice>'."
        )
    return int(digits) / fps


__all__ = [
    "SPLIT_NAMES",
    "extract_frames",
    "extract_all_frames",
    "split_by_clip",
    "write_splits",
    "read_splits",
    "frame_offset_seconds",
]
