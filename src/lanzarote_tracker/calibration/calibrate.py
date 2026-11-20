"""Calibracion pinhole de la webcam.

Practica 2: cv2.calibrateCamera a partir de las correspondencias 3D-2D del
tablero, y cv2.projectPoints para el error de reproyeccion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import CalibrationConfig

Image = np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    """Resultado de la calibracion de la camara.

    Attributes:
        camera_matrix: Matriz intrinseca K, de forma (3, 3).
        distortion_coefficients: Coeficientes de distorsion (k1, k2, p1, p2, k3).
        rms: Error RMS de reproyeccion que devuelve cv2.calibrateCamera.
        image_size: Tamano de las imagenes de calibracion, en (ancho, alto).
        num_images: Numero de imagenes validas usadas.
        rotation_vectors: Vectores de rotacion por imagen.
        translation_vectors: Vectores de traslacion por imagen.
    """

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rms: float
    image_size: Tuple[int, int]
    num_images: int
    rotation_vectors: Tuple[np.ndarray, ...] = ()
    translation_vectors: Tuple[np.ndarray, ...] = ()


def calibrate_camera(
    images: Sequence[Image],
    config: CalibrationConfig,
    recorder: Optional[DebugRecorder] = None,
) -> CalibrationResult:
    """Calibra la camara a partir de fotos del tablero de ajedrez.

    Practica 2: cv2.findChessboardCorners + cv2.cornerSubPix para las esquinas y
    cv2.calibrateCamera para obtener K, los coeficientes de distorsion y el RMS.

    Args:
        images: Fotos del tablero tomadas con la webcam a calibrar.
        config: Configuracion de calibracion (tamano del patron, lado del
            cuadrado, minimo de imagenes).
        recorder: Registro opcional de la secuencia de transformacion.

    Returns:
        El :class:`CalibrationResult` con K, distorsion, RMS y tamano de imagen.

    Raises:
        ValueError: Si no hay suficientes imagenes validas.
    """
    raise NotImplementedError(
        "Pendiente: calibrate_camera (calibration/calibrate.py). "
        "Practica 2: collect_corners + cv2.calibrateCamera."
    )


def reprojection_errors(
    result: CalibrationResult,
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
) -> List[float]:
    """Calcula el error de reproyeccion imagen a imagen.

    Practica 2: cv2.projectPoints con los rvec y tvec de cada imagen y norma L2
    frente a las esquinas detectadas. Sirve para detectar fotos malas y para la
    tabla de resultados del informe.

    Args:
        result: Resultado de la calibracion, con sus rvec y tvec.
        object_points: Puntos 3D del patron por imagen.
        image_points: Esquinas detectadas por imagen.

    Returns:
        Lista con el error medio en pixeles de cada imagen.
    """
    raise NotImplementedError(
        "Pendiente: reprojection_errors (calibration/calibrate.py). "
        "Practica 2: cv2.projectPoints + cv2.norm."
    )


def run_calibration(
    images_dir: Path,
    config: CalibrationConfig,
    recorder: Optional[DebugRecorder] = None,
) -> CalibrationResult:
    """Carga las fotos de un directorio y calibra la camara con ellas.

    Es el punto de entrada que usa `scripts/calibrate_camera.py`.

    Args:
        images_dir: Directorio con las fotos del tablero.
        config: Configuracion de calibracion.
        recorder: Registro opcional de la secuencia de transformacion.

    Returns:
        El resultado de la calibracion.

    Raises:
        FileNotFoundError: Si el directorio no existe o esta vacio.
        ValueError: Si no hay suficientes imagenes validas.
    """
    raise NotImplementedError(
        "Pendiente: run_calibration (calibration/calibrate.py). "
        "Practica 2: load_images + calibrate_camera."
    )


__all__ = [
    "CalibrationResult",
    "calibrate_camera",
    "reprojection_errors",
    "run_calibration",
]
