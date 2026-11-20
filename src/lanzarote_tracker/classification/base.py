"""Interfaz comun de los clasificadores de orientacion del avion.

El sistema clasifica si el avion se ve DE FRENTE (viene hacia la camara) o DE
ESPALDAS (se aleja de la camara). Combinando esa orientacion con el sentido de
uso de la pista se deduce si el avion aterriza o despega; esa logica esta en
:mod:`lanzarote_tracker.classification.arrival_departure`.

Backends:

- ``bow``: bolsa de palabras SIFT + K-means + SVM (Practica 5, apartado A). ES
  LA LINEA BASE y la alternativa clasica.
- ``cnn``: CNN propia. OPCIONAL y PENDIENTE DE APROBACION DEL PROFESOR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder

Image = np.ndarray


@dataclass(frozen=True)
class Orientation:
    """Resultado de clasificar la orientacion de un avion.

    Attributes:
        label: "front" si se ve de frente, "back" si se ve de espaldas.
        confidence: Confianza de la decision, entre 0 y 1.
    """

    label: str
    confidence: float = 0.0

    @property
    def is_front(self) -> bool:
        """True si el avion se ve de frente."""
        return self.label == "front"


class OrientationClassifier(ABC):
    """Interfaz que cumple todo clasificador de orientacion del proyecto."""

    #: Nombre del backend, el mismo que se pone en `config.classification.backend`.
    name: str = "base"

    @abstractmethod
    def fit(self, images: Sequence[Image], labels: Sequence[str]) -> None:
        """Entrena el clasificador.

        Args:
            images: Recortes del avion de la particion de entrenamiento.
            labels: Etiqueta de cada recorte, "front" o "back".

        Raises:
            ValueError: Si no hay ejemplos de las dos clases.
        """

    @abstractmethod
    def predict(
        self,
        image: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Orientation:
        """Clasifica la orientacion de un recorte del avion.

        Args:
            image: Recorte del avion, tal como lo da la caja del seguidor.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            La orientacion con su confianza.

        Raises:
            RuntimeError: Si el clasificador no esta entrenado ni cargado.
        """

    @abstractmethod
    def save(self, directory: Path) -> Tuple[Path, ...]:
        """Guarda el modelo entrenado.

        Los modelos NO se versionan: `models/` esta en .gitignore.

        Args:
            directory: Directorio de destino, normalmente `models/`.

        Returns:
            Las rutas de los ficheros escritos.
        """

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Carga un modelo entrenado previamente.

        Args:
            directory: Directorio donde esta el modelo.

        Raises:
            FileNotFoundError: Si falta algun fichero del modelo. En ese caso
                hay que ejecutar antes `scripts/train_classifier.py`.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """True si el clasificador ya se puede usar para predecir."""


__all__ = ["Orientation", "OrientationClassifier"]
