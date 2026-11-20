"""Backend `coco`: detector preentrenado sobre la clase `airplane`. OPCIONAL.

El profesor ha aprobado el uso de deep learning, pero con una condicion que
manda sobre todo lo demas: **esto sigue siendo un proyecto de vision por
ordenador tradicional**. En consecuencia:

- el backend por defecto sigue siendo `classic`, que no usa deep learning;
- este backend es una ALTERNATIVA con la que comparar en el informe, no el
  camino principal, y no se empieza hasta que `classic` este terminado;
- los frameworks de deep learning NO entran en el entorno base: se instalan
  aparte con ``pip install -e ".[dl]"``.

Fuera de las practicas: la deteccion por redes preentrenadas no se vio en
clase, asi que el modelo elegido hay que justificarlo con bibliografia en el
informe y comparar sus resultados con los del backend `classic`.

CUIDADO con el indice de la clase: hay dos convenciones de etiquetas de COCO y
no coinciden (ver `DetectionConfig.coco_label_convention`). Elegir la
equivocada no da ningun error, simplemente devuelve otra clase.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import COCO_AIRPLANE_CLASS_ID, DetectionConfig
from lanzarote_tracker.detection.base import Detection, PlaneDetector

Image = np.ndarray


class CocoPlaneDetector(PlaneDetector):
    """Detector de aviones basado en un modelo preentrenado en COCO.

    Attributes:
        config: Configuracion de deteccion.
    """

    name = "coco"

    def __init__(self, config: DetectionConfig) -> None:
        """Inicializa el detector y comprueba la coherencia de las etiquetas.

        Args:
            config: Configuracion de deteccion.

        Raises:
            ValueError: Si `coco_label_convention` no es una de las conocidas, o
                si `coco_airplane_class_id` no corresponde a esa convencion.
        """
        expected = COCO_AIRPLANE_CLASS_ID.get(config.coco_label_convention)
        if expected is None:
            raise ValueError(
                f"Convencion de etiquetas COCO desconocida: "
                f"{config.coco_label_convention!r}. "
                f"Disponibles: {', '.join(COCO_AIRPLANE_CLASS_ID)}."
            )
        if config.coco_airplane_class_id != expected:
            raise ValueError(
                f"La convencion {config.coco_label_convention!r} situa la clase "
                f"'airplane' en el indice {expected}, pero la configuracion dice "
                f"{config.coco_airplane_class_id}. Con un indice equivocado el "
                "detector devuelve otra clase sin dar ningun error."
            )
        self.config: DetectionConfig = config

    def detect(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> List[Detection]:
        """Detecta los aviones con el modelo preentrenado.

        Filtrara las cajas por `config.coco_airplane_class_id` (clase
        `airplane`) y por `config.coco_confidence_threshold`.

        Args:
            frame: Fotograma BGR del clip.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            Lista de detecciones ordenada por confianza.

        Raises:
            NotImplementedError: Todavia no implementado. Es una alternativa
                opcional: primero hay que terminar el backend `classic`.
        """
        raise NotImplementedError(
            "El backend de deteccion 'coco' todavia no esta implementado. "
            "Es la alternativa opcional con deep learning: el camino principal "
            "del proyecto es el backend 'classic'. Antes de implementarlo hay "
            "que elegir modelo, instalar el extra con `pip install -e \".[dl]\"` "
            "y fijar config.detection.coco_label_convention."
        )


__all__ = ["CocoPlaneDetector"]
