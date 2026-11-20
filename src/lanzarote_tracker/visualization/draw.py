"""Dibujo de cajas, contornos y etiquetas sobre los fotogramas.

Todas las funciones de este modulo COPIAN la imagen antes de dibujar: las
funciones de dibujo de OpenCV (cv2.rectangle, cv2.putText, cv2.drawContours)
modifican el array que reciben, y si no se copia se estropea el fotograma
original que el resto del pipeline todavia necesita.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from lanzarote_tracker.config import VisualizationConfig

Image = np.ndarray
BoundingBox = Tuple[int, int, int, int]
Contour = np.ndarray


def draw_box(
    frame: Image,
    bbox: BoundingBox,
    label: str = "",
    config: Optional[VisualizationConfig] = None,
    color: Optional[Tuple[int, int, int]] = None,
) -> Image:
    """Dibuja un recuadro, con su etiqueta encima, sobre una copia del fotograma.

    Args:
        frame: Fotograma de entrada, que NO se modifica.
        bbox: Caja (x, y, ancho, alto) en pixeles.
        label: Texto a escribir sobre la caja; vacio para no escribir nada.
        config: Configuracion de visualizacion (grosor, colores, fuente).
        color: Color BGR que sustituye al de la configuracion. Sirve para
            distinguir una caja de deteccion de una de seguimiento.

    Returns:
        Una copia del fotograma con la caja dibujada.
    """
    raise NotImplementedError(
        "Pendiente: draw_box (visualization/draw.py). "
        "frame.copy() + cv2.rectangle + cv2.putText."
    )


def draw_contours(
    frame: Image,
    contours: Sequence[Contour],
    config: Optional[VisualizationConfig] = None,
    color: Optional[Tuple[int, int, int]] = None,
) -> Image:
    """Dibuja contornos sobre una copia del fotograma.

    Args:
        frame: Fotograma de entrada, que NO se modifica.
        contours: Contornos devueltos por cv2.findContours.
        config: Configuracion de visualizacion.
        color: Color BGR que sustituye al de la configuracion.

    Returns:
        Una copia del fotograma con los contornos dibujados.
    """
    raise NotImplementedError(
        "Pendiente: draw_contours (visualization/draw.py). "
        "frame.copy() + cv2.drawContours."
    )


def draw_text(
    frame: Image,
    text: str,
    origin: Tuple[int, int],
    config: Optional[VisualizationConfig] = None,
    color: Optional[Tuple[int, int, int]] = None,
) -> Image:
    """Escribe una linea de texto sobre una copia del fotograma.

    Args:
        frame: Fotograma de entrada, que NO se modifica.
        text: Texto a escribir.
        origin: Esquina inferior izquierda del texto, en pixeles.
        config: Configuracion de visualizacion.
        color: Color BGR que sustituye al de la configuracion.

    Returns:
        Una copia del fotograma con el texto dibujado.
    """
    raise NotImplementedError(
        "Pendiente: draw_text (visualization/draw.py). "
        "frame.copy() + cv2.putText con contorno oscuro para que se lea."
    )


def draw_tracks(
    frame: Image,
    points: np.ndarray,
    previous_points: Optional[np.ndarray] = None,
    config: Optional[VisualizationConfig] = None,
) -> Image:
    """Dibuja los puntos del flujo optico y sus trayectorias.

    Practica 5: es la visualizacion habitual de Lucas-Kanade, con una linea
    entre la posicion anterior y la actual de cada punto.

    Args:
        frame: Fotograma de entrada, que NO se modifica.
        points: Puntos actuales, con forma (N, 1, 2).
        previous_points: Puntos en el fotograma anterior, para las trayectorias.
        config: Configuracion de visualizacion.

    Returns:
        Una copia del fotograma con los puntos y las lineas dibujados.
    """
    raise NotImplementedError(
        "Pendiente: draw_tracks (visualization/draw.py). "
        "frame.copy() + cv2.circle y cv2.line por punto."
    )


__all__ = ["draw_box", "draw_contours", "draw_text", "draw_tracks"]
