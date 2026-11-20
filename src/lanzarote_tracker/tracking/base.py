"""Interfaz comun de los seguidores del avion.

Como la camara del aeropuerto gira y hace zoom, NO se puede usar sustraccion de
fondo: la estrategia del proyecto es detectar cada K fotogramas y seguir al
avion entre detecciones. Esta interfaz cubre la parte de seguimiento; el
gestor que alterna deteccion y seguimiento esta en
:mod:`lanzarote_tracker.tracking.manager`.

Backends (los dos de la Practica 5):

- ``lk``: flujo optico de Lucas-Kanade sobre puntos Shi-Tomasi dentro de la caja.
- ``meanshift``: meanShift sobre el histograma de tono del recorte.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder

Image = np.ndarray
#: Caja delimitadora en pixeles: (x, y, ancho, alto).
BoundingBox = Tuple[int, int, int, int]


class PlaneTracker(ABC):
    """Interfaz que cumple todo seguidor de aviones del proyecto."""

    #: Nombre del backend, el mismo que se pone en `config.tracking.backend`.
    name: str = "base"

    @abstractmethod
    def init(self, frame: Image, bbox: BoundingBox) -> None:
        """Arranca el seguimiento a partir de una deteccion.

        Args:
            frame: Fotograma donde se ha detectado el avion.
            bbox: Caja de la deteccion, en pixeles.

        Raises:
            ValueError: Si la caja esta fuera del fotograma o es degenerada.
        """

    @abstractmethod
    def update(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Optional[BoundingBox]:
        """Sigue al avion en el fotograma siguiente.

        Args:
            frame: Fotograma nuevo.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            La caja actualizada, o None si se ha perdido el objetivo, en cuyo
            caso el gestor fuerza una deteccion nueva.
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """True si el seguidor ya tiene un objetivo asignado."""

    def reset(self) -> None:
        """Olvida el objetivo actual. Por defecto no hace nada."""
        return None


__all__ = ["BoundingBox", "PlaneTracker"]
