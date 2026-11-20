"""Tipos de datos del sistema de seguridad: formas, colores, tarjetas y estados.

Este modulo SI esta implementado: son estructuras de datos, no algoritmos de
vision. El resto del paquete `security` trabaja con ellos.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Shape(str, Enum):
    """Formas admitidas en las tarjetas de desbloqueo.

    Hereda de `str` para poder compararla y serializarla como texto sin
    conversiones, que es comodo en la configuracion y en los tests.
    """

    CIRCLE = "circle"
    TRIANGLE = "triangle"
    SQUARE = "square"
    PENTAGON = "pentagon"


class Color(str, Enum):
    """Colores admitidos en las tarjetas de desbloqueo."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"


class SecurityState(str, Enum):
    """Estados de la maquina de estados del desbloqueo.

    Attributes:
        WAITING: Esperando. No hay ninguna tarjeta delante de la camara y aun
            no se ha empezado a leer, o se acaba de terminar una lectura y se
            espera el hueco antes de la siguiente.
        READING: Leyendo k. Se esta acumulando una lectura consistente de la
            tarjeta numero k.
        CHECKING: Comprobacion. Ya hay `sequence_length` tarjetas leidas y se
            compara la secuencia ENTERA con la esperada.
        UNLOCKED: Desbloqueado. La secuencia era correcta.
        ERROR: Error. La secuencia era incorrecta o se agoto el tiempo. Se
            vuelve a Esperando salvo que se hayan agotado los intentos.
        LOCKED: Bloqueo. Se han encadenado `max_failed_attempts` fallos y el
            sistema no acepta tarjetas durante `lockout_seconds`.
    """

    WAITING = "waiting"
    READING = "reading"
    CHECKING = "checking"
    UNLOCKED = "unlocked"
    ERROR = "error"
    LOCKED = "locked"


#: Caja delimitadora en pixeles: (x, y, ancho, alto).
BoundingBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Card:
    """Una tarjeta reconocida en un fotograma.

    Attributes:
        shape: Forma detectada.
        color: Color detectado.
        bbox: Caja delimitadora del contorno en el fotograma.
        area: Area del contorno en pixeles.
        circularity: Circularidad C = 4*pi*A / P^2 del contorno, entre 0 y 1.
        vertices: Numero de vertices que devolvio cv2.approxPolyDP.
    """

    shape: Shape
    color: Color
    bbox: Optional[BoundingBox] = None
    area: float = 0.0
    circularity: float = 0.0
    vertices: int = 0

    @property
    def key(self) -> Tuple[str, str]:
        """Par (forma, color) que identifica la tarjeta, sin datos geometricos.

        Es lo unico que se compara con la secuencia esperada: dos lecturas de la
        misma tarjeta en fotogramas distintos tienen areas y cajas diferentes.
        """
        return self.shape.value, self.color.value

    def __str__(self) -> str:
        """Representacion legible, p. ej. "circle/red"."""
        return f"{self.shape.value}/{self.color.value}"


__all__ = ["Shape", "Color", "SecurityState", "BoundingBox", "Card"]
