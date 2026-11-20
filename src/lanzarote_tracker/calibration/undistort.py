"""Correccion de la distorsion de la webcam.

Practica 2: cv2.getOptimalNewCameraMatrix + cv2.initUndistortRectifyMap +
cv2.remap. Se precalculan los mapas una sola vez porque la correccion se aplica
a cada fotograma del bloque de seguridad.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from lanzarote_tracker.calibration.calibrate import CalibrationResult

Image = np.ndarray
#: Par de mapas de remapeo (map_x, map_y) que espera cv2.remap.
UndistortMaps = Tuple[np.ndarray, np.ndarray]
#: Rectangulo valido (x, y, ancho, alto) devuelto por getOptimalNewCameraMatrix.
Roi = Tuple[int, int, int, int]


class Undistorter:
    """Aplica la correccion de distorsion con mapas precalculados.

    Practica 2: los mapas se calculan una vez con
    cv2.initUndistortRectifyMap y se reutilizan en cada fotograma con
    cv2.remap, que es mucho mas rapido que llamar a cv2.undistort por fotograma.

    Attributes:
        result: Calibracion de la camara.
        alpha: Parametro de cv2.getOptimalNewCameraMatrix; 0 recorta los bordes
            negros, 1 los conserva todos.
        new_camera_matrix: Matriz K ajustada que devuelve
            getOptimalNewCameraMatrix. Es la que hay que usar para cualquier
            medida sobre la imagen corregida.
        roi: Rectangulo valido de la imagen corregida.
    """

    def __init__(self, result: CalibrationResult, alpha: float = 0.0) -> None:
        """Inicializa el corrector y precalcula los mapas.

        Args:
            result: Resultado de la calibracion con K y los coeficientes.
            alpha: Parametro de recorte de cv2.getOptimalNewCameraMatrix.
        """
        self.result: CalibrationResult = result
        self.alpha: float = float(alpha)

        new_matrix, roi = cv2.getOptimalNewCameraMatrix(
            result.camera_matrix,
            result.distortion_coefficients,
            result.image_size,
            self.alpha,
            result.image_size,
        )
        self.new_camera_matrix: np.ndarray = np.asarray(new_matrix, dtype=np.float64)
        self.roi: Roi = tuple(int(value) for value in roi)  # type: ignore[assignment]

        map_x, map_y = cv2.initUndistortRectifyMap(
            result.camera_matrix,
            result.distortion_coefficients,
            None,
            self.new_camera_matrix,
            result.image_size,
            cv2.CV_16SC2,
        )
        self.maps: UndistortMaps = (map_x, map_y)

    def __call__(self, image: Image, crop: bool = True) -> Image:
        """Corrige la distorsion de un fotograma.

        Args:
            image: Fotograma de entrada, que no se modifica.
            crop: Si se recorta al rectangulo valido (ROI) que devuelve
                cv2.getOptimalNewCameraMatrix.

        Returns:
            Una imagen nueva con la distorsion corregida.

        Raises:
            ValueError: Si el fotograma no tiene el tamano con el que se
                calibro la camara. La calibracion vale para una resolucion
                concreta: no se puede aplicar a otra.
        """
        height, width = image.shape[:2]
        if (width, height) != self.result.image_size:
            expected = self.result.image_size
            raise ValueError(
                f"El fotograma es {width}x{height} y la calibracion se hizo a "
                f"{expected[0]}x{expected[1]}. Usa la misma resolucion que al "
                "calibrar, o vuelve a calibrar con esta."
            )

        undistorted = cv2.remap(image, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
        if crop:
            x, y, roi_width, roi_height = self.roi
            if roi_width > 0 and roi_height > 0:
                undistorted = undistorted[y : y + roi_height, x : x + roi_width]
        return undistorted


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
    height, width = image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        result.camera_matrix,
        result.distortion_coefficients,
        (width, height),
        float(alpha),
        (width, height),
    )
    undistorted = cv2.undistort(
        image,
        result.camera_matrix,
        result.distortion_coefficients,
        None,
        new_matrix,
    )
    if crop:
        x, y, roi_width, roi_height = (int(value) for value in roi)
        if roi_width > 0 and roi_height > 0:
            undistorted = undistorted[y : y + roi_height, x : x + roi_width]
    return undistorted


def side_by_side(original: Image, undistorted: Image) -> Image:
    """Compone la figura antes/despues que pide el informe.

    La imagen corregida suele venir recortada por el ROI, asi que se reescala a
    la altura de la original antes de apilarlas.

    Args:
        original: Fotograma tal cual lo da la camara.
        undistorted: El mismo fotograma con la distorsion corregida.

    Returns:
        Una imagen con las dos vistas una al lado de la otra y sus etiquetas.
    """
    height = original.shape[0]
    scale = height / float(undistorted.shape[0])
    width = max(int(round(undistorted.shape[1] * scale)), 1)
    resized = cv2.resize(undistorted, (width, height), interpolation=cv2.INTER_AREA)

    left, right = original.copy(), resized
    for image, text in ((left, "Original"), (right, "Corregida")):
        cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), thickness=-1)
        cv2.putText(
            image,
            text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.hstack([left, right])


__all__ = ["UndistortMaps", "Roi", "Undistorter", "undistort_image", "side_by_side"]
