"""Guardado y carga de la calibracion de la camara.

La calibracion SI se versiona (``calibration/results/``): es un dato del
hardware concreto con el que se ha hecho el proyecto y el informe la necesita.
Se guarda por duplicado: ``.npz`` para volver a cargarla con NumPy y ``.json``
para poder leerla y pegarla en el informe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from lanzarote_tracker.calibration.calibrate import CalibrationResult
from lanzarote_tracker.config import CalibrationConfig, PathsConfig


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
            fichero.

    Returns:
        Una tupla con las rutas del .npz y del .json escritos.
    """
    raise NotImplementedError(
        "Pendiente: save_calibration (calibration/store.py). "
        "np.savez con camera_matrix, dist_coeffs, rms e image_size, y json.dump "
        "con los mismos valores convertidos a listas."
    )


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
    raise NotImplementedError(
        "Pendiente: load_calibration (calibration/store.py). "
        "np.load del .npz guardado por save_calibration."
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
    raise NotImplementedError(
        "Pendiente: try_load_calibration (calibration/store.py). "
        "load_calibration envuelto en un try/except FileNotFoundError."
    )


__all__ = ["save_calibration", "load_calibration", "try_load_calibration"]
