"""Correccion de la distorsion de la webcam.

Practica 2: cv2.getOptimalNewCameraMatrix + cv2.initUndistortRectifyMap +
cv2.remap. Se precalculan los mapas una sola vez porque la correccion se aplica
a cada fotograma del bloque de seguridad.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from lanzarote_tracker.calibration.calibrate import CalibrationResult

Image = np.ndarray
#: Par de mapas de remapeo (map_x, map_y) que espera cv2.remap.
UndistortMaps = Tuple[np.ndarray, np.ndarray]


class Undistorter:
    """Aplica la correccion de distorsion con mapas precalculados.

    Practica 2: los mapas se calculan una vez con
    cv2.initUndistortRectifyMap y se reutilizan en cada fotograma con
    cv2.remap, que es mucho mas rapido que llamar a cv2.undistort por fotograma.

    Attributes:
        result: Calibracion de la camara.
        alpha: Parametro de cv2.getOptimalNewCameraMatrix; 0 recorta los bordes
            negros, 1 los conserva todos.
    """

    def __init__(self, result: CalibrationResult, alpha: float = 0.0) -> None:
        """Inicializa el corrector y precalcula los mapas.

        Args:
            result: Resultado de la calibracion con K y los coeficientes.
            alpha: Parametro de recorte de cv2.getOptimalNewCameraMatrix.
        """
        raise NotImplementedError(
            "Pendiente: Undistorter.__init__ (calibration/undistort.py). "
            "Practica 2: cv2.getOptimalNewCameraMatrix + cv2.initUndistortRectifyMap."
        )

    def __call__(self, image: Image, crop: bool = True) -> Image:
        """Corrige la distorsion de un fotograma.

        Args:
            image: Fotograma de entrada, que no se modifica.
            crop: Si se recorta al rectangulo valido (ROI) que devuelve
                cv2.getOptimalNewCameraMatrix.

        Returns:
            Una imagen nueva con la distorsion corregida.
        """
        raise NotImplementedError(
            "Pendiente: Undistorter.__call__ (calibration/undistort.py). "
            "Practica 2: cv2.remap con los mapas precalculados."
        )


def undistort_image(
    image: Image,
    result: CalibrationResult,
    alpha: float = 0.0,
    crop: bool = True,
) -> Image:
    """Corrige la distorsion de una sola imagen, sin precalcular mapas.

    Practica 2: cv2.undistort. Para bucles de video usar :class:`Undistorter`,
    que reutiliza los mapas.

    Args:
        image: Imagen de entrada, que no se modifica.
        result: Resultado de la calibracion.
        alpha: Parametro de recorte de cv2.getOptimalNewCameraMatrix.
        crop: Si se recorta al rectangulo valido.

    Returns:
        Una imagen nueva con la distorsion corregida.
    """
    raise NotImplementedError(
        "Pendiente: undistort_image (calibration/undistort.py). "
        "Practica 2: cv2.undistort."
    )


__all__ = ["UndistortMaps", "Undistorter", "undistort_image"]
