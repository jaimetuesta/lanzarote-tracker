"""Guardado y carga de la calibracion de la camara.

La calibracion SI se versiona (``calibration/results/``): es un dato del
hardware concreto con el que se ha hecho el proyecto y el informe la necesita.
Se guarda por duplicado: ``.npz`` para volver a cargarla con NumPy y ``.json``
para poder leerla y pegarla en el informe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from lanzarote_tracker.calibration.calibrate import CalibrationResult
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import CalibrationConfig, PathsConfig

_logger = get_logger(__name__)


def save_calibration(
    result: CalibrationResult,
    paths: PathsConfig,
    config: CalibrationConfig,
) -> Tuple[Path, Path]:
    """Guarda K, los coeficientes de distorsion y el RMS en .npz y .json.

    Args:
        result: Resultado de la calibracion.
        paths: Rutas del proyecto, de las que se toma `calibration_results`.
        config: Configuracion de calibracion, de la que se toman los nombres de
            fichero y los datos del tablero que se anotan en el .json.

    Returns:
        Una tupla con las rutas del .npz y del .json escritos.
    """
    directory = Path(paths.calibration_results)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / config.npz_filename
    json_path = directory / config.json_filename

    np.savez(
        npz_path,
        camera_matrix=result.camera_matrix,
        distortion_coefficients=result.distortion_coefficients,
        rms=np.float64(result.rms),
        image_size=np.array(result.image_size, dtype=np.int32),
        num_images=np.int32(result.num_images),
        rotation_vectors=np.array(result.rotation_vectors, dtype=np.float64),
        translation_vectors=np.array(result.translation_vectors, dtype=np.float64),
    )

    payload = {
        "camera_matrix": result.camera_matrix.tolist(),
        "distortion_coefficients": np.asarray(result.distortion_coefficients)
        .ravel()
        .tolist(),
        "rms": result.rms,
        "image_size": {"width": result.image_size[0], "height": result.image_size[1]},
        "num_images": result.num_images,
        "pattern_size": {
            "columns": config.pattern_size[0],
            "rows": config.pattern_size[1],
        },
        "square_size_mm": config.square_size_mm,
        "calibrated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _logger.info("Calibracion guardada en %s y %s", npz_path, json_path)
    return npz_path, json_path


def load_calibration(
    paths: PathsConfig,
    config: CalibrationConfig,
) -> CalibrationResult:
    """Carga la calibracion guardada previamente.

    Args:
        paths: Rutas del proyecto.
        config: Configuracion de calibracion, de la que se toma el nombre del .npz.

    Returns:
        El :class:`CalibrationResult` reconstruido desde el fichero.

    Raises:
        FileNotFoundError: Si no existe el fichero de calibracion. En ese caso
            hay que ejecutar antes `scripts/calibrate_camera.py`.
    """
    npz_path = Path(paths.calibration_results) / config.npz_filename
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"No existe la calibracion en {npz_path}. Calibra la camara con "
            "`python scripts/calibrate_camera.py`."
        )

    with np.load(npz_path) as data:
        width, height = (int(value) for value in data["image_size"])
        rotations = tuple(np.asarray(item) for item in data["rotation_vectors"])
        translations = tuple(np.asarray(item) for item in data["translation_vectors"])
        return CalibrationResult(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(
                data["distortion_coefficients"], dtype=np.float64
            ),
            rms=float(data["rms"]),
            image_size=(width, height),
            num_images=int(data["num_images"]),
            rotation_vectors=rotations,
            translation_vectors=translations,
        )


def try_load_calibration(
    paths: PathsConfig,
    config: CalibrationConfig,
) -> Optional[CalibrationResult]:
    """Carga la calibracion si existe, y devuelve None si no.

    Util en los pipelines que pueden funcionar sin correccion de distorsion,
    avisando por log de que la calibracion no esta disponible.

    Args:
        paths: Rutas del proyecto.
        config: Configuracion de calibracion.

    Returns:
        El resultado de la calibracion, o None si todavia no se ha calibrado.
    """
    try:
        return load_calibration(paths, config)
    except FileNotFoundError:
        _logger.warning(
            "No hay calibracion guardada: se continua SIN corregir la distorsion. "
            "Calibra la camara con `python scripts/calibrate_camera.py`."
        )
        return None


__all__ = ["save_calibration", "load_calibration", "try_load_calibration"]
