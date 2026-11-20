"""Seguimiento por flujo optico de Lucas-Kanade.

Practica 5: cv2.goodFeaturesToTrack (esquinas de Shi-Tomasi) dentro de la caja
de la deteccion, y cv2.calcOpticalFlowPyrLK para seguir esos puntos de un
fotograma al siguiente. La caja se desplaza con la mediana del movimiento de
los puntos que han sobrevivido, que es mas robusta que la media frente a los
puntos que saltan al fondo.

Es el backend de seguimiento POR DEFECTO: el avion tiene esquinas marcadas
(timon, alas, motores) y el flujo optico aguanta mejor que meanShift los
cambios de escala del zoom de la camara.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import TrackingConfig
from lanzarote_tracker.tracking.base import BoundingBox, PlaneTracker

Image = np.ndarray


class LucasKanadeTracker(PlaneTracker):
    """Seguidor por flujo optico de Lucas-Kanade sobre puntos Shi-Tomasi.

    Attributes:
        config: Configuracion de seguimiento.
    """

    name = "lk"

    def __init__(self, config: TrackingConfig) -> None:
        """Inicializa el seguidor sin objetivo.

        Args:
            config: Configuracion de seguimiento, de la que se toman los
                parametros de Shi-Tomasi y de Lucas-Kanade.
        """
        raise NotImplementedError(
            "Pendiente: LucasKanadeTracker.__init__ (tracking/optical_flow.py)."
        )

    def init(self, frame: Image, bbox: BoundingBox) -> None:
        """Busca puntos Shi-Tomasi dentro de la caja y los toma como objetivo.

        Practica 5: cv2.goodFeaturesToTrack sobre la imagen en escala de grises
        con una mascara que deja pasar solo el interior de la caja, con
        `max_corners`, `quality_level` y `min_distance` de la configuracion.

        Args:
            frame: Fotograma donde se ha detectado el avion.
            bbox: Caja de la deteccion.

        Raises:
            ValueError: Si la caja es degenerada o no se encuentran puntos
                suficientes dentro de ella.
        """
        raise NotImplementedError(
            "Pendiente: LucasKanadeTracker.init (tracking/optical_flow.py). "
            "Practica 5: cv2.goodFeaturesToTrack con mascara de la caja."
        )

    def update(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Optional[BoundingBox]:
        """Sigue los puntos al fotograma siguiente y desplaza la caja.

        Practica 5: cv2.calcOpticalFlowPyrLK entre el fotograma anterior y el
        actual. Se descartan los puntos cuyo `status` sea 0 y, si quedan menos
        de `config.min_tracked_points`, se considera perdido el objetivo.

        Args:
            frame: Fotograma nuevo.
            recorder: Registro opcional de la secuencia de transformacion, al
                que se entrega el fotograma con las trayectorias dibujadas.

        Returns:
            La caja desplazada, o None si se ha perdido el objetivo.
        """
        raise NotImplementedError(
            "Pendiente: LucasKanadeTracker.update (tracking/optical_flow.py). "
            "Practica 5: cv2.calcOpticalFlowPyrLK + mediana del desplazamiento."
        )

    @property
    def is_initialized(self) -> bool:
        """True si hay puntos de seguimiento activos."""
        raise NotImplementedError(
            "Pendiente: LucasKanadeTracker.is_initialized (tracking/optical_flow.py)."
        )

    def reset(self) -> None:
        """Olvida los puntos y el fotograma anterior."""
        raise NotImplementedError(
            "Pendiente: LucasKanadeTracker.reset (tracking/optical_flow.py)."
        )


__all__ = ["LucasKanadeTracker"]
