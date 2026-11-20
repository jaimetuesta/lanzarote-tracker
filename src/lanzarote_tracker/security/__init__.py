"""Sistema de seguridad: desbloqueo mostrando una secuencia de 4 tarjetas.

Cada tarjeta tiene una forma (circulo, triangulo, cuadrado, pentagono) y un
color (rojo, verde, azul, amarillo). La camara es la webcam del usuario, que si
funciona en vivo y es la que se calibra.
"""

from __future__ import annotations

from lanzarote_tracker.security.card_reader import StableCardReader, read_card
from lanzarote_tracker.security.segmentation import (
    build_color_mask,
    clean_mask,
    color_masks,
    largest_contour,
    segment_card,
)
from lanzarote_tracker.security.sequence_fsm import FrameObservation, SequenceFsm
from lanzarote_tracker.security.shape_classifier import (
    approximate_polygon,
    circularity,
    classify_shape,
)
from lanzarote_tracker.security.types import BoundingBox, Card, Color, SecurityState, Shape

__all__ = [
    "StableCardReader",
    "read_card",
    "build_color_mask",
    "clean_mask",
    "color_masks",
    "largest_contour",
    "segment_card",
    "FrameObservation",
    "SequenceFsm",
    "approximate_polygon",
    "circularity",
    "classify_shape",
    "BoundingBox",
    "Card",
    "Color",
    "SecurityState",
    "Shape",
]
