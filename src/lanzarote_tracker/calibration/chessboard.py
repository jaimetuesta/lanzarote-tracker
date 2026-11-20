"""Deteccion del tablero de ajedrez en las fotos de calibracion.

Practica 2: cv2.findChessboardCorners + cv2.cornerSubPix.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import CalibrationConfig

Image = np.ndarray
#: Esquinas detectadas en una imagen, con forma (N, 1, 2) y dtype float32.
Corners = np.ndarray


def find_corners(
    image: Image,
    config: CalibrationConfig,
    refine: bool = True,
) -> Optional[Corners]:
    """Busca las esquinas internas del tablero en una imagen.

    Practica 2: cv2.findChessboardCorners sobre la imagen en escala de grises y,
    si se encuentran, refinado subpixel con cv2.cornerSubPix usando la ventana y
    los criterios de parada de la configuracion.

    Args:
        image: Imagen BGR o en escala de grises con el tablero.
        config: Configuracion de calibracion, de la que se toman `pattern_size`
            y los parametros del refinado subpixel.
        refine: Si se aplica el refinado subpixel.

    Returns:
        Las esquinas con forma (filas*columnas, 1, 2) y dtype float32, o None
        si el tablero no aparece completo en la imagen.
    """
    raise NotImplementedError(
        "Pendiente: find_corners (calibration/chessboard.py). "
        "Practica 2: cv2.findChessboardCorners + cv2.cornerSubPix."
    )


def build_object_points(config: CalibrationConfig) -> np.ndarray:
    """Construye las coordenadas 3D teoricas de las esquinas del tablero.

    El tablero se supone plano con Z = 0, y las esquinas se separan
    `config.square_size_mm` milimetros entre si, de modo que la calibracion
    devuelva magnitudes en milimetros.

    Args:
        config: Configuracion de calibracion, de la que se toman `pattern_size`
            y `square_size_mm`.

    Returns:
        Un array (filas*columnas, 3) de dtype float32 con las coordenadas del
        patron, identico para todas las imagenes.
    """
    raise NotImplementedError(
        "Pendiente: build_object_points (calibration/chessboard.py). "
        "Practica 2: rejilla de puntos con np.mgrid escalada por el lado del cuadrado."
    )


def collect_corners(
    images: Sequence[Image],
    config: CalibrationConfig,
    recorder: Optional[DebugRecorder] = None,
) -> Tuple[List[np.ndarray], List[Corners], Tuple[int, int]]:
    """Recorre las imagenes de calibracion y recoge las correspondencias 3D-2D.

    Practica 2: por cada imagen valida se anade una copia de los puntos 3D del
    patron y las esquinas 2D detectadas. Las imagenes donde no aparece el
    tablero completo se descartan.

    Args:
        images: Fotos del tablero tomadas con la webcam.
        config: Configuracion de calibracion.
        recorder: Registro opcional de la secuencia de transformacion, al que se
            entrega cada imagen con sus esquinas dibujadas
            (cv2.drawChessboardCorners).

    Returns:
        Una tupla (puntos 3D, puntos 2D, tamano de imagen en (ancho, alto)).

    Raises:
        ValueError: Si no se alcanza `config.min_images` imagenes validas.
    """
    raise NotImplementedError(
        "Pendiente: collect_corners (calibration/chessboard.py). "
        "Practica 2: bucle sobre las imagenes con findChessboardCorners."
    )


__all__ = ["Corners", "find_corners", "build_object_points", "collect_corners"]
