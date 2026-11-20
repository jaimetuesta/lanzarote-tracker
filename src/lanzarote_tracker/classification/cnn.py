"""Backend `cnn`: red convolucional propia. DESACTIVADO.

AVISO: este backend NO forma parte del alcance base. Su uso de deep learning
esta PENDIENTE DE APROBACION DEL PROFESOR. Mientras tanto:

- `config.classification.backend` vale "bow" por defecto.
- El entorno NO incluye ningun framework de deep learning; en `environment.yml`
  estan comentados con una nota.
- El constructor falla con un mensaje explicito en vez de importar nada.

Fuera de las practicas: las redes convolucionales no se vieron en las practicas
de la asignatura. Si se aprueba su uso, en el informe hay que justificar la
arquitectura con bibliografia y comparar sus resultados con los del backend
`bow`, que seguira siendo la linea base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from lanzarote_tracker.classification.base import Orientation, OrientationClassifier
from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import ClassificationConfig

Image = np.ndarray

#: Mensaje unico para no repetirlo en cada metodo.
_PENDING_APPROVAL = (
    "El backend de clasificacion 'cnn' esta DESACTIVADO: usa deep learning y su "
    "uso esta pendiente de aprobacion del profesor. Usa el backend 'bow' "
    "(config.classification.backend = 'bow')."
)


class CnnOrientationClassifier(OrientationClassifier):
    """Clasificador de orientacion basado en una CNN propia.

    Attributes:
        config: Configuracion de clasificacion.
    """

    name = "cnn"

    def __init__(self, config: ClassificationConfig) -> None:
        """Rechaza la construccion mientras el backend este pendiente de aprobacion.

        Args:
            config: Configuracion de clasificacion.

        Raises:
            NotImplementedError: Siempre, mientras no se apruebe el uso de deep
                learning.
        """
        raise NotImplementedError(_PENDING_APPROVAL)

    def fit(self, images: Sequence[Image], labels: Sequence[str]) -> None:
        """Entrenaria la CNN con los recortes de entrenamiento.

        Args:
            images: Recortes de entrenamiento.
            labels: Etiqueta de cada recorte.

        Raises:
            NotImplementedError: Siempre, mientras el backend este desactivado.
        """
        raise NotImplementedError(_PENDING_APPROVAL)

    def predict(
        self,
        image: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Orientation:
        """Clasificaria la orientacion con la CNN.

        Args:
            image: Recorte del avion.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            La orientacion y su confianza.

        Raises:
            NotImplementedError: Siempre, mientras el backend este desactivado.
        """
        raise NotImplementedError(_PENDING_APPROVAL)

    def save(self, directory: Path) -> Tuple[Path, ...]:
        """Guardaria los pesos entrenados.

        Args:
            directory: Directorio de destino.

        Returns:
            Las rutas escritas.

        Raises:
            NotImplementedError: Siempre, mientras el backend este desactivado.
        """
        raise NotImplementedError(_PENDING_APPROVAL)

    def load(self, directory: Path) -> None:
        """Cargaria los pesos entrenados.

        Args:
            directory: Directorio donde estan los pesos.

        Raises:
            NotImplementedError: Siempre, mientras el backend este desactivado.
        """
        raise NotImplementedError(_PENDING_APPROVAL)

    @property
    def is_fitted(self) -> bool:
        """Indicaria si la red esta entrenada o cargada.

        Raises:
            NotImplementedError: Siempre, mientras el backend este desactivado.
        """
        raise NotImplementedError(_PENDING_APPROVAL)


__all__ = ["CnnOrientationClassifier"]
