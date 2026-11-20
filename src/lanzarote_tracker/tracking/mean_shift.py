"""Seguimiento por meanShift sobre el histograma de tono.

Practica 5: se calcula el histograma del canal H del recorte inicial
(cv2.calcHist), y en cada fotograma se proyecta ese histograma sobre la imagen
entera (cv2.calcBackProject) y se desplaza la ventana con cv2.meanShift hasta
el maximo de la retroproyeccion.

Es la alternativa a Lucas-Kanade. Funciona bien mientras el avion mantenga su
color frente al fondo, pero meanShift NO cambia el tamano de la ventana, asi
que sufre con el zoom de la camara del aeropuerto. Esa limitacion se compara
con el backend `lk` en el informe.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import TrackingConfig
from lanzarote_tracker.tracking.base import BoundingBox, PlaneTracker

Image = np.ndarray


class MeanShiftTracker(PlaneTracker):
    """Seguidor por meanShift sobre la retroproyeccion del histograma de tono.

    Attributes:
        config: Configuracion de seguimiento.
    """

    name = "meanshift"

    def __init__(self, config: TrackingConfig) -> None:
        """Inicializa el seguidor sin objetivo.

        Args:
            config: Configuracion de seguimiento, de la que se toman el numero
                de bins del histograma y los criterios de parada de meanShift.
        """
        raise NotImplementedError(
            "Pendiente: MeanShiftTracker.__init__ (tracking/mean_shift.py)."
        )

    def init(self, frame: Image, bbox: BoundingBox) -> None:
        """Calcula el histograma de tono del recorte y lo toma como modelo.

        Practica 5: cv2.cvtColor a HSV, cv2.calcHist del canal H dentro de la
        caja con `config.hist_bins` bins, y cv2.normalize del histograma.

        Args:
            frame: Fotograma donde se ha detectado el avion.
            bbox: Caja de la deteccion.

        Raises:
            ValueError: Si la caja es degenerada o cae fuera del fotograma.
        """
        raise NotImplementedError(
            "Pendiente: MeanShiftTracker.init (tracking/mean_shift.py). "
            "Practica 5: cv2.calcHist del canal H + cv2.normalize."
        )

    def update(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Optional[BoundingBox]:
        """Desplaza la ventana al maximo de la retroproyeccion del histograma.

        Practica 5: cv2.calcBackProject con el histograma guardado y
        cv2.meanShift con los criterios de parada de la configuracion.

        Args:
            frame: Fotograma nuevo.
            recorder: Registro opcional de la secuencia de transformacion, al
                que se entrega la imagen de retroproyeccion.

        Returns:
            La caja desplazada, o None si meanShift no converge.
        """
        raise NotImplementedError(
            "Pendiente: MeanShiftTracker.update (tracking/mean_shift.py). "
            "Practica 5: cv2.calcBackProject + cv2.meanShift."
        )

    @property
    def is_initialized(self) -> bool:
        """True si hay un histograma modelo y una ventana activa."""
        raise NotImplementedError(
            "Pendiente: MeanShiftTracker.is_initialized (tracking/mean_shift.py)."
        )

    def reset(self) -> None:
        """Olvida el histograma modelo y la ventana."""
        raise NotImplementedError(
            "Pendiente: MeanShiftTracker.reset (tracking/mean_shift.py)."
        )


__all__ = ["MeanShiftTracker"]
