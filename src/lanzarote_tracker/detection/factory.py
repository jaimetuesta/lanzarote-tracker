"""Fabrica de detectores: elige el backend desde la configuracion."""

from __future__ import annotations

from lanzarote_tracker.config import DetectionConfig
from lanzarote_tracker.detection.base import PlaneDetector
from lanzarote_tracker.detection.classic import ClassicPlaneDetector
from lanzarote_tracker.detection.coco import CocoPlaneDetector


def create_detector(config: DetectionConfig) -> PlaneDetector:
    """Construye el detector indicado en `config.backend`.

    Args:
        config: Configuracion de deteccion.

    Returns:
        Una instancia de :class:`~lanzarote_tracker.detection.base.PlaneDetector`.

    Raises:
        ValueError: Si el backend no esta en `config.available_backends`, o si
            la convencion de etiquetas del backend "coco" es incoherente.
    """
    backend = config.backend.lower()
    if backend not in config.available_backends:
        raise ValueError(
            f"Backend de deteccion desconocido: {config.backend!r}. "
            f"Disponibles: {', '.join(config.available_backends)}."
        )
    if backend == "classic":
        return ClassicPlaneDetector(config)
    return CocoPlaneDetector(config)


__all__ = ["create_detector"]
