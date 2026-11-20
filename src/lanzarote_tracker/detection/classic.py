"""Backend `classic`: deteccion del avion por color y morfologia.

ES EL BACKEND POR DEFECTO del proyecto, y el unico que se implementa en el
alcance base: no usa deep learning, asi que no depende de la aprobacion del
profesor.

Idea del pipeline, por fotograma:

1. Conversion a HSV (Practica 1: cv2.cvtColor).
2. Mascara del cielo por rango HSV y su negado, para quedarse con lo que NO es
   cielo (Practica 1: cv2.inRange + cv2.bitwise_not).
3. Apertura y cierre morfologicos para quitar ruido y cerrar el fuselaje
   (Practica 3: cv2.morphologyEx).
4. Contornos y filtrado por area y relacion de aspecto (Practica 4:
   cv2.findContours + cv2.contourArea + cv2.boundingRect).

NOTA: la camara del aeropuerto gira y hace zoom, asi que NO se usa sustraccion
de fondo (cv2.createBackgroundSubtractorMOG2 y similares): el fondo cambia
continuamente y la sustraccion marcaria la escena entera como movimiento.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import DetectionConfig
from lanzarote_tracker.detection.base import Detection, PlaneDetector

Image = np.ndarray


class ClassicPlaneDetector(PlaneDetector):
    """Detector del avion por segmentacion de color y morfologia.

    Attributes:
        config: Configuracion de deteccion.
    """

    name = "classic"

    def __init__(self, config: DetectionConfig) -> None:
        """Inicializa el detector.

        Args:
            config: Configuracion de deteccion, de la que se toman el rango HSV
                del cielo, las areas limite y los parametros morfologicos.
        """
        self.config: DetectionConfig = config

    def detect(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> List[Detection]:
        """Detecta los aviones del fotograma por color y morfologia.

        Args:
            frame: Fotograma BGR del clip.
            recorder: Registro opcional de la secuencia de transformacion, al
                que se entregan el fotograma, la imagen HSV, la mascara del
                cielo, su negado, la mascara limpia y los contornos dibujados.

        Returns:
            Lista de detecciones ordenada de mayor a menor area.
        """
        raise NotImplementedError(
            "Pendiente: ClassicPlaneDetector.detect (detection/classic.py). "
            "Practicas 1, 3 y 4: HSV + inRange del cielo + bitwise_not + "
            "morfologia + findContours filtrando por area y relacion de aspecto."
        )


__all__ = ["ClassicPlaneDetector"]
