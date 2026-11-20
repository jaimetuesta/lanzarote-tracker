"""Fabrica de clasificadores: elige el backend desde la configuracion."""

from __future__ import annotations

from lanzarote_tracker.classification.base import OrientationClassifier
from lanzarote_tracker.classification.bow import BowOrientationClassifier
from lanzarote_tracker.classification.cnn import CnnOrientationClassifier
from lanzarote_tracker.config import ClassificationConfig


def create_classifier(config: ClassificationConfig) -> OrientationClassifier:
    """Construye el clasificador indicado en `config.backend`.

    Args:
        config: Configuracion de clasificacion.

    Returns:
        Una instancia de
        :class:`~lanzarote_tracker.classification.base.OrientationClassifier`.

    Raises:
        ValueError: Si el backend no esta en `config.available_backends`.
        NotImplementedError: Si se pide el backend "cnn", que esta desactivado
            a la espera de la aprobacion del profesor.
    """
    backend = config.backend.lower()
    if backend not in config.available_backends:
        raise ValueError(
            f"Backend de clasificacion desconocido: {config.backend!r}. "
            f"Disponibles: {', '.join(config.available_backends)}."
        )
    if backend == "bow":
        return BowOrientationClassifier(config)
    return CnnOrientationClassifier(config)


__all__ = ["create_classifier"]
