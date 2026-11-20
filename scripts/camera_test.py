"""Prueba de la webcam: muestra el video con el FPS dibujado en cada fotograma.

Este script SI funciona de verdad: es la comprobacion de que la camara del
bloque de seguridad esta accesible y de que el contador de FPS es correcto.

Para la prueba del directo de YouTube, ver `scripts/stream_test.py`.

Uso::

    python scripts/camera_test.py
    python scripts/camera_test.py --camera 1 --no-mirror
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import cv2

from lanzarote_tracker.common.fps import FpsMeter, draw_fps
from lanzarote_tracker.common.io_utils import is_quit_key, open_camera
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import AppConfig, get_config

_logger = get_logger(__name__)


def run(config: AppConfig) -> int:
    """Abre la webcam y muestra el video con el FPS incrustado.

    Sale con 'q' o ESC.

    Args:
        config: Configuracion de la aplicacion.

    Returns:
        0 si termina correctamente, 1 si la camara no se puede abrir.
    """
    try:
        capture = open_camera(config.camera)
    except RuntimeError as error:
        _logger.error("%s", error)
        return 1

    meter = FpsMeter(window=config.visualization.fps_window)
    window_name = f"{config.camera.window_name} - camara ('q' o ESC para salir)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.camera.width, config.camera.height)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                _logger.warning("Fotograma perdido; se cierra la camara.")
                break

            if config.camera.mirror:
                frame = cv2.flip(frame, 1)

            fps = meter.tick()
            # draw_fps copia el fotograma: el original queda intacto.
            annotated = draw_fps(frame, fps, config.visualization)
            cv2.imshow(window_name, annotated)

            if is_quit_key(cv2.waitKey(1) & 0xFF, config.camera.quit_keys):
                break
    except KeyboardInterrupt:
        _logger.info("Interrumpido por el usuario.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    """Punto de entrada del script.

    Returns:
        El codigo de salida del proceso.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Abre la webcam y muestra el video con el FPS dibujado en cada "
            "fotograma. Sale con 'q' o ESC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config = get_config()
    parser.add_argument(
        "--camera",
        type=int,
        default=config.camera.index,
        help="Indice de la webcam.",
    )
    parser.add_argument(
        "--width", type=int, default=config.camera.width, help="Ancho solicitado."
    )
    parser.add_argument(
        "--height", type=int, default=config.camera.height, help="Alto solicitado."
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="No voltear la imagen horizontalmente.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(
        config,
        camera=replace(
            config.camera,
            index=args.camera,
            width=args.width,
            height=args.height,
            mirror=not args.no_mirror,
        ),
    )
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
