"""Gestor de deteccion periodica + seguimiento entre detecciones.

La camara del aeropuerto gira y hace zoom, asi que la sustraccion de fondo no
sirve y detectar en todos los fotogramas es caro. La estrategia del proyecto es:

- detectar cada K fotogramas (`TrackingConfig.detect_every_k_frames`);
- entre detecciones, seguir al avion con el backend elegido;
- forzar una deteccion antes de tiempo si el seguimiento se pierde
  `max_frames_lost` fotogramas seguidos.

Este gestor es la pieza que une :mod:`lanzarote_tracker.detection` con
:mod:`lanzarote_tracker.tracking` y la que consume `scripts/run_tracker.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import TrackingConfig
from lanzarote_tracker.detection.base import PlaneDetector
from lanzarote_tracker.tracking.base import BoundingBox, PlaneTracker

Image = np.ndarray


@dataclass(frozen=True)
class TrackState:
    """Resultado del gestor en un fotograma.

    Attributes:
        bbox: Caja del avion en este fotograma, o None si no hay objetivo.
        from_detection: True si la caja viene de una deteccion y False si viene
            del seguimiento. Sirve para pintarlas de colores distintos en el
            video de entrega y para las estadisticas del informe.
        frame_index: Indice del fotograma dentro del clip.
        lost_frames: Fotogramas seguidos sin objetivo valido.
    """

    bbox: Optional[BoundingBox]
    from_detection: bool
    frame_index: int
    lost_frames: int


class TrackingManager:
    """Alterna deteccion periodica y seguimiento entre detecciones.

    Attributes:
        detector: Detector de aviones.
        tracker: Seguidor entre detecciones.
        config: Configuracion de seguimiento, de la que se toma K.
    """

    def __init__(
        self,
        detector: PlaneDetector,
        tracker: PlaneTracker,
        config: TrackingConfig,
    ) -> None:
        """Inicializa el gestor sin objetivo.

        Args:
            detector: Detector construido por
                :func:`lanzarote_tracker.detection.factory.create_detector`.
            tracker: Seguidor construido por
                :func:`lanzarote_tracker.tracking.factory.create_tracker`.
            config: Configuracion de seguimiento.
        """
        raise NotImplementedError(
            "Pendiente: TrackingManager.__init__ (tracking/manager.py)."
        )

    def update(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> TrackState:
        """Procesa un fotograma: detecta o sigue, segun toque.

        Detecta cuando el indice del fotograma es multiplo de
        `config.detect_every_k_frames`, cuando todavia no hay objetivo o cuando
        el seguimiento lleva `config.max_frames_lost` fotogramas perdido. En el
        resto de fotogramas sigue.

        Args:
            frame: Fotograma del clip.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            El :class:`TrackState` de este fotograma.
        """
        raise NotImplementedError(
            "Pendiente: TrackingManager.update (tracking/manager.py). "
            "Deteccion cada K fotogramas y seguimiento en el resto."
        )

    def reset(self) -> None:
        """Olvida el objetivo y reinicia el contador de fotogramas."""
        raise NotImplementedError(
            "Pendiente: TrackingManager.reset (tracking/manager.py)."
        )


__all__ = ["TrackState", "TrackingManager"]
