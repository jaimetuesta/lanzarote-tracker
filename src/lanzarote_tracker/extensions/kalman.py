"""AMPLIACION 1 (NO BASE): filtro de Kalman para suavizar la trayectoria.

Que hara: mantener un modelo de estado del avion (posicion y velocidad en la
imagen) y, en cada fotograma, predecir donde deberia estar antes de medir. Con
eso el sistema podria:

- seguir dando una caja plausible en los fotogramas en los que la deteccion o
  el seguimiento fallan, en vez de perder el objetivo;
- suavizar el salto de la caja al pasar de seguimiento a deteccion;
- restringir la busqueda del detector a los alrededores de la prediccion.

OpenCV trae `cv2.KalmanFilter`, con lo que no hacen falta dependencias nuevas.

Fuera de las practicas: el filtro de Kalman no se vio en las practicas de la
asignatura. Si se implementa, hay que justificarlo con bibliografia en el
informe y comparar la trayectoria con y sin filtro.
"""

from __future__ import annotations

from typing import Optional, Tuple

BoundingBox = Tuple[int, int, int, int]


class KalmanBoxFilter:
    """Filtro de Kalman sobre el centro y el tamano de la caja del avion."""

    def __init__(self, dt: float = 1.0) -> None:
        """Inicializara el filtro con un modelo de velocidad constante.

        Args:
            dt: Intervalo entre fotogramas, en unidades de fotograma.

        Raises:
            NotImplementedError: Siempre: es una ampliacion, no alcance base.
        """
        raise NotImplementedError(
            "AMPLIACION no implementada: filtro de Kalman (extensions/kalman.py). "
            "No forma parte del alcance base del proyecto."
        )

    def predict(self) -> BoundingBox:
        """Predecira la caja del fotograma siguiente.

        Returns:
            La caja predicha.

        Raises:
            NotImplementedError: Siempre: es una ampliacion.
        """
        raise NotImplementedError(
            "AMPLIACION no implementada: filtro de Kalman (extensions/kalman.py)."
        )

    def correct(self, bbox: Optional[BoundingBox]) -> BoundingBox:
        """Corregira la prediccion con la medida del detector o del seguidor.

        Args:
            bbox: Caja medida, o None si en este fotograma no hay medida.

        Returns:
            La caja corregida.

        Raises:
            NotImplementedError: Siempre: es una ampliacion.
        """
        raise NotImplementedError(
            "AMPLIACION no implementada: filtro de Kalman (extensions/kalman.py)."
        )


__all__ = ["KalmanBoxFilter"]
