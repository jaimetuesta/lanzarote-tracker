"""Calibra la webcam a partir de las fotos del tablero de `calibration/images/`.

Guarda K, los coeficientes de distorsion y el RMS en `calibration/results/`,
en .npz (para volver a cargarlos) y en .json (para pegarlos en el informe).

ESTADO: la calibracion todavia no esta implementada (Practica 2). El script
arranca y falla con un mensaje que dice que falta.

Uso::

    python scripts/calibrate_camera.py
    python scripts/calibrate_camera.py --images calibration/images --debug
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lanzarote_tracker.calibration.calibrate import run_calibration
from lanzarote_tracker.calibration.store import save_calibration
from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import get_config

_logger = get_logger(__name__)


def main() -> int:
    """Punto de entrada del script.

    Returns:
        0 si la calibracion termina, 1 si falla o falta por implementar.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calibra la webcam con las fotos del tablero y guarda K, los "
            "coeficientes de distorsion y el RMS en calibration/results/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config = get_config()
    parser.add_argument(
        "--images",
        type=Path,
        default=config.paths.calibration_images,
        help="Directorio con las fotos del tablero.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Guarda la secuencia de transformacion en results/calibration/.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(config, debug=replace(config.debug, enabled=args.debug))
    recorder = DebugRecorder("calibration", config.debug)

    try:
        result = run_calibration(args.images, config.calibration, recorder=recorder)
        npz_path, json_path = save_calibration(result, config.paths, config.calibration)
    except (NotImplementedError, FileNotFoundError, ValueError) as error:
        _logger.error("%s", error)
        return 1

    recorder.save_all()
    _logger.info("RMS de reproyeccion: %.4f px", result.rms)
    _logger.info("Guardado en %s y %s", npz_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
