"""Calibracion de la webcam: tablero, modelo pinhole y correccion de distorsion.

Practica 2 de la asignatura. La calibracion se aplica a la webcam, que es la
camara del bloque de seguridad; la camara del aeropuerto no es nuestra y no se
puede calibrar.
"""

from __future__ import annotations

from lanzarote_tracker.calibration.calibrate import (
    CalibrationResult,
    aggregate_rms,
    calibrate_camera,
    reprojection_errors,
    run_calibration,
)
from lanzarote_tracker.calibration.chessboard import (
    build_object_points,
    collect_corners,
    detect_pattern_size,
    find_corners,
    parse_pattern_size,
)
from lanzarote_tracker.calibration.store import (
    load_calibration,
    save_calibration,
    try_load_calibration,
)
from lanzarote_tracker.calibration.undistort import (
    Undistorter,
    side_by_side,
    undistort_image,
)

__all__ = [
    "CalibrationResult",
    "aggregate_rms",
    "calibrate_camera",
    "reprojection_errors",
    "run_calibration",
    "build_object_points",
    "collect_corners",
    "detect_pattern_size",
    "find_corners",
    "parse_pattern_size",
    "load_calibration",
    "save_calibration",
    "try_load_calibration",
    "Undistorter",
    "side_by_side",
    "undistort_image",
]
