"""Interfaz comun de los detectores de avion.

Hay dos backends detras de esta interfaz:

- ``classic``: segmentacion propia del avion frente al cielo y la pista, por
  color y morfologia. ES EL BACKEND POR DEFECTO.
- ``coco``: detector preentrenado sobre la clase ``airplane``. OPCIONAL y
  DESACTIVADO, porque su uso de deep learning esta PENDIENTE DE APROBACION DEL
  PROFESOR.

El resto del sistema solo conoce esta interfaz, de modo que cambiar de backend
es cambiar un valor en `config.detection.backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder

Image = np.ndarray
#: Caja delimitadora en pixeles: (x, y, ancho, alto).
BoundingBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """Un avion detectado en un fotograma.

    Attributes:
        bbox: Caja delimitadora (x, y, ancho, alto) en pixeles.
        confidence: Confianza de la deteccion, entre 0 y 1. El backend
            ``classic`` no da una probabilidad: usa una medida derivada del area
            y la forma del contorno, documentada en su modulo.
        area: Area en pixeles del contorno o de la caja.
    """

    bbox: BoundingBox
    confidence: float = 1.0
    area: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        """Centro de la caja, en pixeles."""
        x, y, width, height = self.bbox
        return x + width / 2.0, y + height / 2.0


class PlaneDetector(ABC):
    """Interfaz que cumple todo detector de aviones del proyecto."""

    #: Nombre del backend, el mismo que se pone en `config.detection.backend`.
    name: str = "base"

    @abstractmethod
    def detect(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> List[Detection]:
        """Detecta los aviones presentes en un fotograma.

        Args:
            frame: Fotograma BGR del clip.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            Lista de detecciones, ordenada de mayor a menor confianza. Vacia si
            no hay ningun avion.
        """

    def detect_best(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Optional[Detection]:
        """Devuelve la mejor deteccion del fotograma, o None si no hay ninguna.

        En esta camara solo se sigue a un avion a la vez, asi que casi todo el
        sistema usa este metodo y no :meth:`detect`.

        Args:
            frame: Fotograma BGR del clip.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            La deteccion de mayor confianza, o None.
        """
        detections = self.detect(frame, recorder=recorder)
        return detections[0] if detections else None


__all__ = ["BoundingBox", "Detection", "PlaneDetector"]
