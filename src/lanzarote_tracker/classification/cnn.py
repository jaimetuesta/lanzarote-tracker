"""Backend `cnn`: red convolucional propia. OPCIONAL.

El profesor ha aprobado el uso de deep learning, pero con una condicion que
manda sobre todo lo demas: **esto sigue siendo un proyecto de vision por
ordenador tradicional**. En consecuencia:

- el backend por defecto sigue siendo `bow` (SIFT + K-means + SVM), que es la
  LINEA BASE del proyecto;
- la CNN es la ALTERNATIVA con la que comparar en el informe, no el camino
  principal, y no se empieza hasta que `bow` este terminado y medido: sin la
  linea base no hay con que comparar, y la comparacion es justamente lo que
  hace interesante a la CNN en la memoria;
- los frameworks de deep learning NO entran en el entorno base: se instalan
  aparte con ``pip install -e ".[dl]"``.

Fuera de las practicas: las redes convolucionales no se vieron en las practicas
de la asignatura. Hay que justificar la arquitectura con bibliografia en el
informe.

Aviso sobre el dataset: con clips propios el conjunto sera pequeno, asi que lo
razonable es partir de una red preentrenada y reentrenar solo las ultimas capas
(transfer learning) en vez de entrenar desde cero. La division sigue siendo POR
CLIPS COMPLETOS: con una CNN, dos fotogramas casi identicos repartidos entre
entrenamiento y validacion inflan las metricas todavia mas que con `bow`.
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
_NOT_IMPLEMENTED = (
    "El backend de clasificacion 'cnn' todavia no esta implementado. Es la "
    "alternativa opcional con deep learning: la linea base del proyecto es el "
    "backend 'bow' (SIFT + K-means + SVM), que debe estar terminado y medido "
    "antes, para poder comparar. Instala el extra con `pip install -e \".[dl]\"`."
)


class CnnOrientationClassifier(OrientationClassifier):
    """Clasificador de orientacion basado en una CNN propia.

    Attributes:
        config: Configuracion de clasificacion.
    """

    name = "cnn"

    def __init__(self, config: ClassificationConfig) -> None:
        """Inicializa el clasificador sin entrenar.

        No importa aqui ningun framework de deep learning: el import va dentro
        de :meth:`fit` y :meth:`load`, para que construir el objeto (y por tanto
        importar el paquete y pasar los tests) no exija tener instalado el extra.

        Args:
            config: Configuracion de clasificacion.
        """
        self.config: ClassificationConfig = config

    def fit(self, images: Sequence[Image], labels: Sequence[str]) -> None:
        """Entrenara la CNN con los recortes de entrenamiento.

        Args:
            images: Recortes de entrenamiento.
            labels: Etiqueta de cada recorte, "front" o "back".

        Raises:
            NotImplementedError: Todavia no implementado.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict(
        self,
        image: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Orientation:
        """Clasificara la orientacion con la CNN.

        Args:
            image: Recorte del avion.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            La orientacion y su confianza.

        Raises:
            NotImplementedError: Todavia no implementado.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def save(self, directory: Path) -> Tuple[Path, ...]:
        """Guardara los pesos entrenados en `models/`, que no se versiona.

        Args:
            directory: Directorio de destino.

        Returns:
            Las rutas escritas.

        Raises:
            NotImplementedError: Todavia no implementado.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def load(self, directory: Path) -> None:
        """Cargara los pesos entrenados.

        Args:
            directory: Directorio donde estan los pesos.

        Raises:
            NotImplementedError: Todavia no implementado.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @property
    def is_fitted(self) -> bool:
        """True si la red esta entrenada o cargada.

        Returns:
            De momento siempre False: la red no se puede entrenar todavia.
        """
        return False


__all__ = ["CnnOrientationClassifier"]
