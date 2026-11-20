"""Backend `bow`: bolsa de palabras visuales SIFT + K-means + SVM.

Practica 5, apartado A. ES LA LINEA BASE del proyecto y la alternativa clasica
a la CNN. El pipeline es el de la practica:

1. Descriptores SIFT de cada recorte de entrenamiento
   (cv2.SIFT_create().detectAndCompute).
2. Vocabulario visual: K-means sobre todos los descriptores
   (sklearn.cluster.KMeans con `config.vocabulary_size` clusters).
3. Histograma de palabras visuales por imagen, normalizado.
4. SVM sobre esos histogramas (sklearn.svm.SVC).

SIFT esta en el modulo principal de OpenCV desde la version 4.4, asi que NO
hace falta opencv-contrib-python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from lanzarote_tracker.classification.base import Orientation, OrientationClassifier
from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import ClassificationConfig

Image = np.ndarray


class BowOrientationClassifier(OrientationClassifier):
    """Clasificador de orientacion por bolsa de palabras visuales.

    Attributes:
        config: Configuracion de clasificacion.
    """

    name = "bow"

    def __init__(self, config: ClassificationConfig) -> None:
        """Inicializa el clasificador sin entrenar.

        Args:
            config: Configuracion de clasificacion, de la que se toman el
                tamano del vocabulario y los parametros del SVM.
        """
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.__init__ (classification/bow.py)."
        )

    def extract_descriptors(self, image: Image) -> Optional[np.ndarray]:
        """Extrae los descriptores SIFT de un recorte.

        Practica 5: el recorte se pasa a escala de grises y se reescala a
        `config.patch_size` antes de cv2.SIFT_create().detectAndCompute, para
        que el zoom de la camara no cambie el numero de descriptores.

        Args:
            image: Recorte del avion.

        Returns:
            Un array (N, 128) con los descriptores, o None si SIFT no encuentra
            ningun punto de interes en el recorte.
        """
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.extract_descriptors "
            "(classification/bow.py). Practica 5: cv2.SIFT_create + detectAndCompute."
        )

    def build_vocabulary(self, descriptors: Sequence[np.ndarray]) -> np.ndarray:
        """Construye el vocabulario visual agrupando los descriptores.

        Practica 5: sklearn.cluster.KMeans con `config.vocabulary_size`
        clusters y `config.random_state` para que el resultado sea reproducible.

        Args:
            descriptors: Descriptores SIFT de todas las imagenes de entrenamiento.

        Returns:
            Los centroides del vocabulario, con forma (vocabulary_size, 128).
        """
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.build_vocabulary "
            "(classification/bow.py). Practica 5: sklearn.cluster.KMeans."
        )

    def build_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """Convierte los descriptores de una imagen en su histograma de palabras.

        Practica 5: cada descriptor se asigna a la palabra visual mas cercana y
        se cuenta; el histograma se normaliza para que no dependa del numero de
        puntos de interes detectados.

        Args:
            descriptors: Descriptores SIFT de una imagen.

        Returns:
            El histograma normalizado, con forma (vocabulary_size,).
        """
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.build_histogram "
            "(classification/bow.py). Practica 5: asignacion al centroide mas cercano."
        )

    def fit(self, images: Sequence[Image], labels: Sequence[str]) -> None:
        """Entrena el vocabulario y el SVM.

        Practica 5: descriptores SIFT de todas las imagenes, K-means para el
        vocabulario, histograma por imagen y sklearn.svm.SVC sobre esos
        histogramas con el kernel y la C de la configuracion.

        Args:
            images: Recortes de entrenamiento.
            labels: Etiqueta de cada recorte, "front" o "back".

        Raises:
            ValueError: Si falta alguna de las dos clases o no hay descriptores.
        """
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.fit (classification/bow.py). "
            "Practica 5 apartado A: SIFT + K-means + SVM."
        )

    def predict(
        self,
        image: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Orientation:
        """Clasifica la orientacion de un recorte.

        Args:
            image: Recorte del avion.
            recorder: Registro opcional de la secuencia de transformacion, al
                que se entregan el recorte y sus puntos SIFT dibujados
                (cv2.drawKeypoints).

        Returns:
            La orientacion y su confianza, tomada de `decision_function` del SVM.

        Raises:
            RuntimeError: Si el clasificador no esta entrenado ni cargado.
        """
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.predict (classification/bow.py)."
        )

    def save(self, directory: Path) -> Tuple[Path, ...]:
        """Guarda el vocabulario y el SVM en `models/`.

        Args:
            directory: Directorio de destino.

        Returns:
            Las rutas del vocabulario y del clasificador.
        """
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.save (classification/bow.py). "
            "np.savez del vocabulario + joblib.dump del SVM."
        )

    def load(self, directory: Path) -> None:
        """Carga el vocabulario y el SVM guardados previamente.

        Args:
            directory: Directorio donde esta el modelo.

        Raises:
            FileNotFoundError: Si falta el vocabulario o el clasificador.
        """
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.load (classification/bow.py)."
        )

    @property
    def is_fitted(self) -> bool:
        """True si hay vocabulario y SVM listos."""
        raise NotImplementedError(
            "Pendiente: BowOrientationClassifier.is_fitted (classification/bow.py)."
        )


__all__ = ["BowOrientationClassifier"]
