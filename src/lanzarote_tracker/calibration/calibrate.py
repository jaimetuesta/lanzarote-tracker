"""Calibracion pinhole de la webcam.

Practica 2: cv2.calibrateCamera a partir de las correspondencias 3D-2D del
tablero, y cv2.projectPoints para el error de reproyeccion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from lanzarote_tracker.calibration.chessboard import collect_corners
from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.common.io_utils import load_images
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import CalibrationConfig

Image = np.ndarray

_logger = get_logger(__name__)


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

    @property
    def focal_length(self) -> Tuple[float, float]:
        """Distancias focales (fx, fy) en pixeles."""
        return float(self.camera_matrix[0, 0]), float(self.camera_matrix[1, 1])

    @property
    def principal_point(self) -> Tuple[float, float]:
        """Punto principal (cx, cy) en pixeles."""
        return float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2])

    def summary(self) -> str:
        """Devuelve un resumen legible de la calibracion, para el log y el informe.

        Returns:
            Un texto de varias lineas con fx, fy, cx, cy, los coeficientes y el RMS.
        """
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        coefficients = np.asarray(self.distortion_coefficients).ravel()
        names = ("k1", "k2", "p1", "p2", "k3")
        formatted = "  ".join(
            f"{name}={value: .6f}" for name, value in zip(names, coefficients)
        )
        return (
            f"Imagenes validas: {self.num_images}\n"
            f"Tamano de imagen:  {self.image_size[0]}x{self.image_size[1]}\n"
            f"fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}\n"
            f"Distorsion: {formatted}\n"
            f"RMS de reproyeccion: {self.rms:.4f} px"
        )


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
    object_points, image_points, image_size = collect_corners(
        images, config, recorder=recorder
    )
    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    result = CalibrationResult(
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion_coefficients=np.asarray(distortion, dtype=np.float64),
        rms=float(rms),
        image_size=image_size,
        num_images=len(object_points),
        rotation_vectors=tuple(rotations),
        translation_vectors=tuple(translations),
    )

    if result.rms > config.max_acceptable_rms:
        _logger.warning(
            "El RMS de reproyeccion es %.4f px, por encima del maximo aceptable "
            "(%.2f px). Repite las fotos con el tablero pegado a algo rigido y "
            "llevandolo a las esquinas del encuadre.",
            result.rms,
            config.max_acceptable_rms,
        )
    return result


def reprojection_errors(
    result: CalibrationResult,
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
) -> List[float]:
    """Calcula el RMS de reproyeccion imagen a imagen.

    Practica 2: cv2.projectPoints con los rvec y tvec de cada imagen y norma L2
    frente a las esquinas detectadas. Sirve para detectar fotos malas y para la
    tabla de resultados del informe.

    NORMALIZACION: la norma L2 del residuo se divide por la RAIZ del numero de
    esquinas, no por el numero de esquinas. Asi cada valor es un RMS en pixeles
    y la media cuadratica de todos ellos coincide EXACTAMENTE con el RMS que
    devuelve cv2.calibrateCamera (ver :func:`aggregate_rms`), de modo que los
    dos numeros del informe son comparables.

    El tutorial de OpenCV divide por el numero de esquinas, lo que da un valor
    sqrt(N) veces menor que el RMS global: con 54 esquinas por tablero, un RMS
    de 2.87 px aparece como 0.35 px. No se usa aqui para no mezclar dos
    estadisticos distintos en la misma tabla.

    Args:
        result: Resultado de la calibracion, con sus rvec y tvec.
        object_points: Puntos 3D del patron por imagen.
        image_points: Esquinas detectadas por imagen.

    Returns:
        Lista con el RMS de reproyeccion en pixeles de cada imagen.

    Raises:
        ValueError: Si el resultado no trae los vectores de rotacion y
            traslacion, o si el numero de imagenes no coincide.
    """
    if not result.rotation_vectors or not result.translation_vectors:
        raise ValueError(
            "El resultado de la calibracion no trae los vectores de rotacion y "
            "traslacion, asi que no se puede calcular el error por imagen."
        )
    if not (
        len(object_points) == len(image_points) == len(result.rotation_vectors)
    ):
        raise ValueError(
            "El numero de imagenes no coincide entre los puntos 3D, los puntos "
            "2D y los vectores de la calibracion."
        )

    errors: List[float] = []
    for points_3d, points_2d, rotation, translation in zip(
        object_points, image_points, result.rotation_vectors, result.translation_vectors
    ):
        projected, _ = cv2.projectPoints(
            points_3d,
            rotation,
            translation,
            result.camera_matrix,
            result.distortion_coefficients,
        )
        error = cv2.norm(points_2d, projected, cv2.NORM_L2) / np.sqrt(len(projected))
        errors.append(float(error))
    return errors


def aggregate_rms(errors: Sequence[float]) -> float:
    """Combina los RMS por imagen en el RMS global de la calibracion.

    Es la media CUADRATICA, no la aritmetica: con los valores que devuelve
    :func:`reprojection_errors` y tableros de igual numero de esquinas, el
    resultado coincide con el RMS de cv2.calibrateCamera.

    Args:
        errors: RMS por imagen, en pixeles.

    Returns:
        El RMS global en pixeles, o 0.0 si no hay imagenes.
    """
    if not errors:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=np.float64)))))


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
    images_dir = Path(images_dir)
    images = load_images(images_dir)
    if not images:
        raise FileNotFoundError(
            f"No hay imagenes en {images_dir}. Toma fotos del tablero con "
            "`python scripts/capture_calibration_images.py`."
        )
    _logger.info("Calibrando con %d fotos de %s", len(images), images_dir)
    return calibrate_camera(images, config, recorder=recorder)


__all__ = [
    "CalibrationResult",
    "aggregate_rms",
    "calibrate_camera",
    "reprojection_errors",
    "run_calibration",
]
