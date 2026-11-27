"""Clasificacion de la forma de una tarjeta a partir de su contorno.

Practica 4: cv2.approxPolyDP sobre el contorno y conteo de vertices para
triangulo (3), cuadrado (4) y pentagono (5). El circulo NO se distingue por
numero de vertices, porque approxPolyDP le asigna un numero inestable: se
detecta por su circularidad

    C = 4 * pi * A / P^2

que vale 1 para un circulo perfecto y baja al alejarse de el (A es el area del
contorno y P su perimetro).

ORDEN DE LAS COMPROBACIONES, y por que. Medido sobre las tarjetas generadas:

    forma       circularidad   vertices
    triangulo       0.566          3
    cuadrado        0.791          4
    pentagono       0.785          5
    circulo         0.897          8

Dos cosas saltan a la vista. La primera es que la circularidad NO separa
cuadrado de pentagono: 0.791 y 0.785 son el mismo numero a efectos practicos, y
por eso hacen falta los vertices. La segunda es que el circulo no mide 1.0 sino
0.897, porque cv2.arcLength recorre el borde en escalera y sobreestima el
perimetro de una curva.

De ahi que los vertices se comprueben PRIMERO y la circularidad solo decida
cuando no salen 3, 4 ni 5. Si se mirase la circularidad antes, el cuadrado
(0.791) quedaria a nueve milesimas del umbral y bastaria un poco de desenfoque
de la webcam para que se leyera como un circulo. Separados por vertices el
margen es de varias unidades enteras, no de milesimas.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
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
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return 0.0
    return float(4.0 * math.pi * cv2.contourArea(contour) / (perimeter * perimeter))


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
    epsilon = config.approx_poly_epsilon_ratio * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def is_square(contour: Contour, config: SecurityConfig) -> bool:
    """Comprueba que un contorno de cuatro vertices sea cuadrado y no rectangular.

    Practica 4: cv2.boundingRect y relacion de aspecto.

    Args:
        contour: Contorno de cuatro vertices.
        config: Configuracion de seguridad, de la que se toma la tolerancia.

    Returns:
        True si el ancho y el alto se parecen lo bastante.
    """
    _, _, width, height = cv2.boundingRect(contour)
    if height <= 0:
        return False
    ratio = width / float(height)
    return abs(ratio - 1.0) <= config.square_aspect_tolerance


def classify_shape(contour: Contour, config: SecurityConfig) -> Optional[Shape]:
    """Decide que forma tiene el contorno de una tarjeta.

    Orden de decision (ver el porque en el docstring del modulo):

    1. Se cuentan los vertices de `approximate_polygon` y se buscan en
       `config.vertices_by_shape`: 3 triangulo, 4 cuadrado, 5 pentagono.
    2. Con 4 vertices se comprueba ademas la relacion de aspecto de la caja,
       para no confundir un cuadrado con un rectangulo alargado del fondo.
    3. Si no salen 3, 4 ni 5 vertices, se mira la circularidad: por encima de
       `config.circularity_threshold` es un CIRCLE.

    Args:
        contour: Contorno de la tarjeta.
        config: Configuracion de seguridad.

    Returns:
        La forma reconocida, o None si no encaja en ninguna de las cuatro.
    """
    vertices = len(approximate_polygon(contour, config))

    for name, expected in config.vertices_by_shape.items():
        if vertices != expected:
            continue
        shape = Shape(name)
        if shape is Shape.SQUARE and not is_square(contour, config):
            return None
        return shape

    if circularity(contour) >= config.circularity_threshold:
        return Shape.CIRCLE
    return None


__all__ = [
    "circularity",
    "approximate_polygon",
    "is_square",
    "classify_shape",
]
