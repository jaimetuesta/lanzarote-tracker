"""Calibracion de la webcam: tablero, modelo pinhole y correccion de distorsion.

Practica 2 de la asignatura. La calibracion se aplica a la webcam, que es la
camara del bloque de seguridad; la camara del aeropuerto no es nuestra y no se
puede calibrar.
"""

from __future__ import annotations

from lanzarote_tracker.calibration.calibrate import (
    CalibrationResult,
    calibrate_camera,
    reprojection_errors,
    run_calibration,
)
from lanzarote_tracker.calibration.chessboard import (
    build_object_points,
    collect_corners,
    find_corners,
)
from lanzarote_tracker.calibration.store import (
    load_calibration,
    save_calibration,
    try_load_calibration,
)
from lanzarote_tracker.calibration.undistort import Undistorter, undistort_image

__all__ = [
    "CalibrationResult",
    "calibrate_camera",
    "reprojection_errors",
    "run_calibration",
    "build_object_points",
    "collect_corners",
    "find_corners",
    "load_calibration",
    "save_calibration",
    "try_load_calibration",
    "Undistorter",
    "undistort_image",
]
