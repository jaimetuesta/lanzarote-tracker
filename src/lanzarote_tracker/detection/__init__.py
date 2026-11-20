"""Deteccion del avion: interfaz comun y backends classic (por defecto) y coco."""

from __future__ import annotations

from lanzarote_tracker.detection.base import BoundingBox, Detection, PlaneDetector
from lanzarote_tracker.detection.classic import ClassicPlaneDetector
from lanzarote_tracker.detection.coco import CocoPlaneDetector
from lanzarote_tracker.detection.factory import create_detector

__all__ = [
    "BoundingBox",
    "Detection",
    "PlaneDetector",
    "ClassicPlaneDetector",
    "CocoPlaneDetector",
    "create_detector",
]
