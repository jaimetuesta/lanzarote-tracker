"""Captura fotos del tablero de ajedrez con la webcam, para calibrar.

Abre la webcam y guarda una foto en `calibration/images/` cada vez que se pulsa
ESPACIO. Sale con 'q' o ESC.

Consejos para una buena calibracion (Practica 2): tomar al menos
`CalibrationConfig.min_images` fotos, con el tablero en posiciones e
inclinaciones distintas, ocupando zonas diferentes del encuadre y llegando
hasta los bordes, que es donde mas se nota la distorsion.

Uso::

    python scripts/capture_calibration_images.py
    python scripts/capture_calibration_images.py --output calibration/images --camera 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2

from lanzarote_tracker.common.fps import FpsMeter, draw_fps
from lanzarote_tracker.common.io_utils import is_quit_key, open_camera, write_image
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import AppConfig, get_config

_logger = get_logger(__name__)

#: Tecla que dispara la captura de una foto.
CAPTURE_KEY = ord(" ")


def run(config: AppConfig, output_dir: Path) -> int:
    """Muestra la webcam y guarda una foto cada vez que se pulsa ESPACIO.

    Args:
        config: Configuracion de la aplicacion.
        output_dir: Directorio donde se guardan las fotos.

    Returns:
        0 si termina correctamente, 1 si la camara no se puede abrir.
    """
    try:
        capture = open_camera(config.camera)
    except RuntimeError as error:
        _logger.error("%s", error)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = len(list(output_dir.glob("calib_*.png")))
    meter = FpsMeter(window=config.visualization.fps_window)
    window_name = "Calibracion - ESPACIO para capturar, 'q' o ESC para salir"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.camera.width, config.camera.height)
    _logger.info("Guardando en %s. Ya habia %d fotos.", output_dir, saved)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                _logger.warning("Fotograma perdido; se cierra la camara.")
                break

            # El fotograma que se GUARDA es el original, sin voltear ni anotar:
            # la calibracion debe hacerse sobre la imagen tal cual la da la camara.
            annotated = draw_fps(frame, meter.tick(), config.visualization)
            cv2.putText(
                annotated,
                f"Fotos: {saved} (minimo {config.calibration.min_images})",
                (config.visualization.card_margin, annotated.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.visualization.font_scale,
                config.visualization.text_color,
                config.visualization.font_thickness,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if is_quit_key(key, config.camera.quit_keys):
                break
            if key == CAPTURE_KEY:
                path = write_image(output_dir / f"calib_{saved:03d}.png", frame)
                saved += 1
                _logger.info("Guardada %s (%d en total).", path.name, saved)
    except KeyboardInterrupt:
        _logger.info("Interrumpido por el usuario.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if saved < config.calibration.min_images:
        _logger.warning(
            "Solo hay %d fotos y la calibracion pide al menos %d.",
            saved,
            config.calibration.min_images,
        )
    return 0


def main() -> int:
    """Punto de entrada del script.

    Returns:
        El codigo de salida del proceso.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Abre la webcam y guarda una foto del tablero de ajedrez cada vez "
            "que se pulsa ESPACIO. Sale con 'q' o ESC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config = get_config()
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.calibration_images,
        help="Directorio donde se guardan las fotos.",
    )
    parser.add_argument(
        "--camera", type=int, default=config.camera.index, help="Indice de la webcam."
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(config, camera=replace(config.camera, index=args.camera))
    return run(config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
