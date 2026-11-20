"""Backend `coco`: detector preentrenado sobre la clase `airplane`. DESACTIVADO.

AVISO: este backend NO forma parte del alcance base. Usa un modelo de deep
learning preentrenado y su uso esta PENDIENTE DE APROBACION DEL PROFESOR. Por
eso:

- `config.detection.backend` vale "classic" por defecto.
- El entorno NO incluye ningun framework de deep learning; en `environment.yml`
  estan comentados con una nota.
- El constructor falla con un mensaje explicito en vez de intentar cargar pesos.

Fuera de las practicas: la deteccion por redes preentrenadas no se vio en
clase. Si se aprueba su uso, en el informe hay que justificar el modelo elegido
con bibliografia y comparar sus resultados con los del backend `classic`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import DetectionConfig
from lanzarote_tracker.detection.base import Detection, PlaneDetector

Image = np.ndarray

#: Mensaje unico para no repetirlo en cada metodo.
_PENDING_APPROVAL = (
    "El backend de deteccion 'coco' esta DESACTIVADO: usa deep learning y su "
    "uso esta pendiente de aprobacion del profesor. Usa el backend 'classic' "
    "(config.detection.backend = 'classic')."
)


class CocoPlaneDetector(PlaneDetector):
    """Detector de aviones basado en un modelo preentrenado en COCO.

    Attributes:
        config: Configuracion de deteccion.
    """

    name = "coco"

    def __init__(self, config: DetectionConfig) -> None:
        """Rechaza la construccion mientras el backend este pendiente de aprobacion.

        Args:
            config: Configuracion de deteccion.

        Raises:
            NotImplementedError: Siempre, mientras no se apruebe el uso de deep
                learning.
        """
        raise NotImplementedError(_PENDING_APPROVAL)

    def detect(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> List[Detection]:
        """Detectaria los aviones con el modelo preentrenado.

        Filtraria las cajas por `config.coco_airplane_class_id` (clase
        `airplane` de COCO) y por `config.coco_confidence_threshold`.

        Args:
            frame: Fotograma BGR del clip.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            Lista de detecciones ordenada por confianza.

        Raises:
            NotImplementedError: Siempre, mientras el backend este desactivado.
        """
        raise NotImplementedError(_PENDING_APPROVAL)


__all__ = ["CocoPlaneDetector"]
