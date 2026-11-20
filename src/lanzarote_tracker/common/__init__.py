"""Utilidades comunes: entrada/salida, contador de FPS, logging y modo depuracion."""

from __future__ import annotations

from lanzarote_tracker.common.debug import DebugRecorder, to_bgr
from lanzarote_tracker.common.fps import FpsMeter, draw_fps
from lanzarote_tracker.common.io_utils import (
    load_image,
    load_images,
    open_camera,
    open_video,
    show_image,
    video_properties,
    write_image,
)
from lanzarote_tracker.common.log import get_logger, setup_logging

__all__ = [
    "DebugRecorder",
    "to_bgr",
    "FpsMeter",
    "draw_fps",
    "load_image",
    "load_images",
    "open_camera",
    "open_video",
    "show_image",
    "video_properties",
    "write_image",
    "get_logger",
    "setup_logging",
]
