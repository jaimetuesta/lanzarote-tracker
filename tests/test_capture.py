"""Tests de la captura: clips, metadatos, fotogramas y division del dataset.

No se toca la red: los clips se generan con `cv2.VideoWriter` sobre fotogramas
sinteticos, que es exactamente lo que escribe el grabador.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from lanzarote_tracker.capture.frames import (
    SPLIT_NAMES,
    extract_frames,
    frame_offset_seconds,
    read_splits,
    split_by_clip,
    write_splits,
)
from lanzarote_tracker.capture.recorder import (
    ClipMetadata,
    build_clip_name,
    clip_time_utc,
    load_clip_metadata,
    metadata_path_for,
)
from lanzarote_tracker.config import CaptureConfig, ClassificationConfig, get_config


@pytest.fixture()
def capture_config() -> CaptureConfig:
    """Configuracion de captura por defecto."""
    return get_config().capture


def make_clip(path: Path, frames: int, fps: float = 25.0, size=(64, 48)) -> Path:
    """Genera un clip sintetico con un numero conocido de fotogramas.

    Args:
        path: Ruta del clip a escribir.
        frames: Numero de fotogramas.
        fps: FPS del clip.
        size: Tamano (ancho, alto).

    Returns:
        La ruta del clip escrito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "No se pudo crear el clip de prueba"
    try:
        for index in range(frames):
            frame = np.full((size[1], size[0], 3), index % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


# --- Nombres y metadatos ----------------------------------------------------
def test_clip_name_carries_the_utc_timestamp(capture_config: CaptureConfig) -> None:
    """El nombre del clip lleva su hora de inicio en UTC."""
    moment = datetime(2026, 11, 27, 17, 30, 0, tzinfo=timezone.utc)
    assert build_clip_name(moment, capture_config) == "clip_20261127T173000Z.mp4"


def test_clip_name_converts_to_utc(capture_config: CaptureConfig) -> None:
    """Una hora con otra zona horaria se convierte a UTC, no se copia tal cual."""
    from datetime import timedelta

    madrid = timezone(timedelta(hours=1))
    moment = datetime(2026, 11, 27, 18, 30, 0, tzinfo=madrid)
    assert build_clip_name(moment, capture_config) == "clip_20261127T173000Z.mp4"


def test_metadata_path_sits_next_to_the_clip(capture_config: CaptureConfig) -> None:
    """El JSON acompana al clip con el mismo nombre."""
    path = metadata_path_for(Path("data/clips/clip_20261127T173000Z.mp4"), capture_config)
    assert path.name == "clip_20261127T173000Z.json"


def make_metadata(tmp_path: Path, **overrides) -> ClipMetadata:
    """Construye unos metadatos de prueba.

    Args:
        tmp_path: Directorio temporal.
        **overrides: Campos a sobreescribir.

    Returns:
        Los metadatos.
    """
    started = datetime(2026, 11, 27, 17, 30, 0, tzinfo=timezone.utc)
    defaults = dict(
        clip_path=tmp_path / "clip_20261127T173000Z.mp4",
        started_at_utc=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        started_at_unix=int(started.timestamp()),
        duration_seconds=60.0,
        fps=25.0,
        effective_fps=25.0,
        width=1920,
        height=1080,
        frame_count=1500,
        source_url="https://www.youtube.com/watch?v=test",
    )
    defaults.update(overrides)
    return ClipMetadata(**defaults)


def test_metadata_round_trip(tmp_path: Path, capture_config: CaptureConfig) -> None:
    """Los metadatos escritos se vuelven a leer igual."""
    metadata = make_metadata(tmp_path)
    json_path = metadata_path_for(metadata.clip_path, capture_config)
    json_path.write_text(
        json.dumps(metadata.to_json(), indent=2), encoding="utf-8"
    )
    loaded = load_clip_metadata(metadata.clip_path, capture_config)
    assert loaded == metadata


def test_missing_metadata_is_reported(
    tmp_path: Path, capture_config: CaptureConfig
) -> None:
    """Sin JSON no se sabe cuando se grabo: hay que avisar, no seguir."""
    with pytest.raises(FileNotFoundError, match="marca de tiempo"):
        load_clip_metadata(tmp_path / "clip_sin_json.mp4", capture_config)


def test_clip_time_adds_the_offset(tmp_path: Path) -> None:
    """La hora de un fotograma es la del clip mas su desplazamiento.

    Es lo que hace falta para cruzar un fotograma concreto con las fichas: no
    vale la hora de inicio si el avion aparece en el segundo 40.
    """
    metadata = make_metadata(tmp_path)
    moment = clip_time_utc(metadata, offset_seconds=40.0)
    assert moment == datetime(2026, 11, 27, 17, 30, 40, tzinfo=timezone.utc)


# --- Extraccion de fotogramas -----------------------------------------------
def test_extract_frames_respects_the_step(
    tmp_path: Path, capture_config: CaptureConfig
) -> None:
    """Se guarda uno de cada `frame_step` fotogramas."""
    clip = make_clip(tmp_path / "clips" / "clip_a.mp4", frames=20)
    config = replace(capture_config, frame_step=5)
    written = extract_frames(clip, tmp_path / "frames", config)
    assert len(written) == 4


def test_extracted_frames_go_to_a_folder_per_clip(
    tmp_path: Path, capture_config: CaptureConfig
) -> None:
    """Cada clip tiene su subdirectorio: de ahi depende dividir por clips."""
    clip = make_clip(tmp_path / "clips" / "clip_a.mp4", frames=10)
    config = replace(capture_config, frame_step=5)
    written = extract_frames(clip, tmp_path / "frames", config)
    assert all(path.parent.name == "clip_a" for path in written)


def test_frame_names_carry_the_clip_index(
    tmp_path: Path, capture_config: CaptureConfig
) -> None:
    """El nombre lleva el indice DENTRO DEL CLIP, no el del conjunto extraido.

    Es lo que permite reconstruir en que segundo aparece cada fotograma.
    """
    clip = make_clip(tmp_path / "clips" / "clip_a.mp4", frames=10)
    config = replace(capture_config, frame_step=5)
    written = extract_frames(clip, tmp_path / "frames", config)
    assert [path.stem for path in written] == ["frame_000000", "frame_000005"]


def test_extract_frames_rejects_a_missing_clip(
    tmp_path: Path, capture_config: CaptureConfig
) -> None:
    """Un clip que no existe se avisa."""
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "no_existe.mp4", tmp_path / "frames", capture_config)


def test_extract_frames_rejects_a_zero_step(
    tmp_path: Path, capture_config: CaptureConfig
) -> None:
    """Un paso de 0 seria un bucle infinito o una division por cero."""
    clip = make_clip(tmp_path / "clips" / "clip_a.mp4", frames=4)
    with pytest.raises(ValueError):
        extract_frames(clip, tmp_path / "frames", replace(capture_config, frame_step=0))


def test_frame_offset_in_seconds() -> None:
    """El indice del fotograma y los FPS dan el segundo del clip."""
    assert frame_offset_seconds("frame_000250.png", fps=25.0, frame_step=5) == 10.0


def test_frame_offset_must_use_the_effective_fps(tmp_path: Path) -> None:
    """Con los FPS declarados en vez de los reales, el instante sale mal.

    El directo declara 60 FPS y entrega unos 22. Un fotograma del segundo 12
    se situaria en el 4.5, suficiente para cruzarlo con la ficha equivocada.
    """
    metadata = make_metadata(tmp_path, fps=60.0, effective_fps=22.4)
    con_declarados = frame_offset_seconds("frame_000269.png", metadata.fps, 1)
    con_reales = frame_offset_seconds("frame_000269.png", metadata.effective_fps, 1)
    assert con_declarados == pytest.approx(4.48, abs=0.01)
    assert con_reales == pytest.approx(12.0, abs=0.1)


def test_frame_offset_rejects_a_bad_name() -> None:
    """Un nombre sin indice no se puede situar en el tiempo."""
    with pytest.raises(ValueError):
        frame_offset_seconds("imagen.png", fps=25.0, frame_step=5)


# --- Division del dataset ---------------------------------------------------
@pytest.fixture()
def classification_config() -> ClassificationConfig:
    """Configuracion de clasificacion por defecto."""
    return get_config().classification


def test_split_assigns_every_clip_exactly_once(
    classification_config: ClassificationConfig,
) -> None:
    """Ningun clip se pierde ni se repite entre particiones."""
    names = [f"clip_{index:02d}" for index in range(20)]
    splits = split_by_clip(names, classification_config)
    assigned = [name for partition in SPLIT_NAMES for name in splits[partition]]
    assert sorted(assigned) == sorted(names)
    assert len(assigned) == len(set(assigned))


def test_split_never_cuts_a_clip_between_partitions(
    classification_config: ClassificationConfig,
) -> None:
    """Un clip entero cae en una sola particion.

    Es la regla que protege las metricas: dos fotogramas del mismo clip son
    casi identicos.
    """
    names = [f"clip_{index:02d}" for index in range(20)]
    splits = split_by_clip(names, classification_config)
    for first in SPLIT_NAMES:
        for second in SPLIT_NAMES:
            if first < second:
                assert not set(splits[first]) & set(splits[second])


def test_split_is_reproducible(classification_config: ClassificationConfig) -> None:
    """La misma semilla da el mismo reparto."""
    names = [f"clip_{index:02d}" for index in range(20)]
    assert split_by_clip(names, classification_config) == split_by_clip(
        names, classification_config
    )


def test_a_different_seed_changes_the_split(
    classification_config: ClassificationConfig,
) -> None:
    """Cambiar la semilla cambia el reparto."""
    names = [f"clip_{index:02d}" for index in range(20)]
    other = replace(classification_config, random_state=7)
    assert split_by_clip(names, classification_config) != split_by_clip(names, other)


def test_validation_and_test_are_never_empty(
    classification_config: ClassificationConfig,
) -> None:
    """Con pocos clips, el redondeo tiende a dejar val y test vacios.

    Si eso pasa no se puede medir nada, asi que se les garantiza un clip.
    """
    names = ["clip_a", "clip_b", "clip_c"]
    splits = split_by_clip(names, classification_config)
    for partition in SPLIT_NAMES:
        assert splits[partition], f"La particion '{partition}' ha quedado vacia"


def test_split_needs_at_least_one_clip_per_partition(
    classification_config: ClassificationConfig,
) -> None:
    """Con menos clips que particiones no hay division posible."""
    with pytest.raises(ValueError, match="al menos"):
        split_by_clip(["clip_a", "clip_b"], classification_config)


def test_split_rejects_ratios_that_do_not_add_up(
    classification_config: ClassificationConfig,
) -> None:
    """Unas proporciones que no suman 1 son un error de configuracion."""
    broken = replace(classification_config, split_ratios=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="suman"):
        split_by_clip([f"clip_{i}" for i in range(10)], broken)


def test_split_ignores_duplicate_names(
    classification_config: ClassificationConfig,
) -> None:
    """Un clip repetido en la entrada no acaba en dos particiones."""
    names = [f"clip_{index:02d}" for index in range(10)] * 2
    splits = split_by_clip(names, classification_config)
    assigned = [name for partition in SPLIT_NAMES for name in splits[partition]]
    assert len(assigned) == len(set(assigned)) == 10


def test_splits_round_trip(
    tmp_path: Path, classification_config: ClassificationConfig
) -> None:
    """La division escrita se vuelve a leer igual."""
    paths = replace(get_config().paths, splits=tmp_path)
    names = [f"clip_{index:02d}" for index in range(12)]
    splits = split_by_clip(names, classification_config)
    write_splits(splits, paths)
    assert read_splits(paths) == splits


def test_reading_missing_splits_is_reported(tmp_path: Path) -> None:
    """Sin particiones escritas se dice como crearlas."""
    paths = replace(get_config().paths, splits=tmp_path / "vacio")
    with pytest.raises(FileNotFoundError, match="build_dataset"):
        read_splits(paths)
