"""Clasificacion de la forma de una tarjeta a partir de su contorno.

Practica 4: cv2.approxPolyDP sobre el contorno y conteo de vertices para
triangulo (3), cuadrado (4) y pentagono (5). El circulo NO se distingue por
numero de vertices, porque approxPolyDP le asigna un numero inestable: se
detecta por su circularidad

    C = 4 * pi * A / P^2

que vale 1 para un circulo perfecto y baja al alejarse de el (A es el area del
contorno y P su perimetro).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from lanzarote_tracker.config import SecurityConfig
from lanzarote_tracker.security.types import Shape

Contour = np.ndarray


def circularity(contour: Contour) -> float:
    """Calcula la circularidad C = 4*pi*A / P^2 de un contorno.

    Practica 4: cv2.contourArea y cv2.arcLength.

    Args:
        contour: Contorno devuelto por cv2.findContours.

    Returns:
        La circularidad, entre 0 y 1 aproximadamente. Vale 1 para un circulo
        perfecto y 0.0 si el perimetro es nulo.
    """
    raise NotImplementedError(
        "Pendiente: circularity (security/shape_classifier.py). "
        "Practica 4: 4*pi*cv2.contourArea / cv2.arcLength**2."
    )


def approximate_polygon(contour: Contour, config: SecurityConfig) -> Contour:
    """Aproxima el contorno por un poligono con menos vertices.

    Practica 4: cv2.approxPolyDP con epsilon igual a
    `config.approx_poly_epsilon_ratio` por el perimetro del contorno
    (cv2.arcLength), que es la forma habitual de hacer el epsilon
    independiente del tamano de la figura en la imagen.

    Args:
        contour: Contorno de entrada.
        config: Configuracion de seguridad, de la que se toma el ratio de epsilon.

    Returns:
        El poligono aproximado, con la misma forma que un contorno.
    """
    raise NotImplementedError(
        "Pendiente: approximate_polygon (security/shape_classifier.py). "
        "Practica 4: cv2.approxPolyDP con epsilon = ratio * cv2.arcLength."
    )


def classify_shape(contour: Contour, config: SecurityConfig) -> Optional[Shape]:
    """Decide que forma tiene el contorno de una tarjeta.

    Orden de decision:

    1. Si la circularidad supera `config.circularity_threshold`, es un CIRCLE.
    2. Si no, se cuenta el numero de vertices de `approximate_polygon` y se
       busca en `config.vertices_by_shape`: 3 triangulo, 4 cuadrado, 5 pentagono.
    3. Con 4 vertices se comprueba ademas la relacion de aspecto de la caja
       (cv2.boundingRect) contra `config.square_aspect_tolerance`, para no
       confundir un cuadrado con un rectangulo alargado del fondo.

    Args:
        contour: Contorno de la tarjeta.
        config: Configuracion de seguridad.

    Returns:
        La forma reconocida, o None si no encaja en ninguna de las cuatro.
    """
    raise NotImplementedError(
        "Pendiente: classify_shape (security/shape_classifier.py). "
        "Practica 4: circularidad para el circulo y numero de vertices de "
        "cv2.approxPolyDP para el resto."
    )


__all__ = ["circularity", "approximate_polygon", "classify_shape"]
