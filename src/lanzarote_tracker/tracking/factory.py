"""Fabrica de seguidores: elige el backend desde la configuracion."""

from __future__ import annotations

from lanzarote_tracker.config import TrackingConfig
from lanzarote_tracker.tracking.base import PlaneTracker
from lanzarote_tracker.tracking.mean_shift import MeanShiftTracker
from lanzarote_tracker.tracking.optical_flow import LucasKanadeTracker


def create_tracker(config: TrackingConfig) -> PlaneTracker:
    """Construye el seguidor indicado en `config.backend`.

    Args:
        config: Configuracion de seguimiento.

    Returns:
        Una instancia de :class:`~lanzarote_tracker.tracking.base.PlaneTracker`.

    Raises:
        ValueError: Si el backend no esta en `config.available_backends`.
    """
    backend = config.backend.lower()
    if backend not in config.available_backends:
        raise ValueError(
            f"Backend de seguimiento desconocido: {config.backend!r}. "
            f"Disponibles: {', '.join(config.available_backends)}."
        )
    if backend == "lk":
        return LucasKanadeTracker(config)
    return MeanShiftTracker(config)


__all__ = ["create_tracker"]
