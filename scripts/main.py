"""Flujo completo: desbloqueo con la webcam y despues el tracker sobre clips.

ESTADO ACTUAL DEL ESQUELETO: el bloque de seguridad y el tracker todavia no
estan implementados, asi que por defecto este script hace lo unico que ya
funciona de verdad: abre la webcam y muestra el video con el FPS dibujado en
cada fotograma, hasta que se pulsa 'q' o ESC.

Con `--security` o `--tracker` se intenta ejecutar el bloque correspondiente, y
el script falla con un mensaje claro diciendo que falta por implementar.

Uso::

    python scripts/main.py
    python scripts/main.py --security
    python scripts/main.py --tracker data/clips/clip_20261120T173000Z.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2

from lanzarote_tracker.common.fps import FpsMeter, draw_fps
from lanzarote_tracker.common.io_utils import is_quit_key, open_camera
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import AppConfig, get_config

_logger = get_logger(__name__)


def run_preview(config: AppConfig) -> int:
    """Muestra la webcam con el FPS incrustado hasta que el usuario sale.

    Es el unico tramo del flujo completo que ya funciona de verdad.

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
    window_name = f"{config.camera.window_name} ('q' o ESC para salir)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.camera.width, config.camera.height)
    _logger.info("Mostrando la webcam. Pulsa 'q' o ESC para salir.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                _logger.warning("Fotograma perdido; se cierra la camara.")
                break

            if config.camera.mirror:
                frame = cv2.flip(frame, 1)

            annotated = draw_fps(frame, meter.tick(), config.visualization)
            cv2.imshow(window_name, annotated)

            if is_quit_key(cv2.waitKey(1) & 0xFF, config.camera.quit_keys):
                break
    except KeyboardInterrupt:
        _logger.info("Interrumpido por el usuario.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def run_security(config: AppConfig) -> int:
    """Ejecutara el desbloqueo por secuencia de tarjetas con la webcam.

    Args:
        config: Configuracion de la aplicacion.

    Returns:
        0 si el usuario se desbloquea.

    Raises:
        NotImplementedError: Mientras el bloque de seguridad no este implementado.
    """
    raise NotImplementedError(
        "El bloque de seguridad todavia no esta implementado. "
        "Faltan: security/segmentation.py, security/shape_classifier.py, "
        "security/card_reader.py y security/sequence_fsm.py. "
        "Ejecuta `python scripts/main.py` sin opciones para la vista previa de "
        "la webcam, que si funciona."
    )


def run_tracker(config: AppConfig, clip_path: Path) -> int:
    """Ejecutara deteccion, seguimiento, clasificacion y ficha sobre un clip.

    Args:
        config: Configuracion de la aplicacion.
        clip_path: Clip grabado sobre el que trabajar.

    Returns:
        0 si termina correctamente.

    Raises:
        NotImplementedError: Mientras el tracker no este implementado.
    """
    raise NotImplementedError(
        "El tracker todavia no esta implementado. "
        "Faltan: detection/classic.py, tracking/optical_flow.py, "
        "tracking/manager.py, classification/bow.py y visualization/hud.py. "
        "Ejecuta `python scripts/main.py` sin opciones para la vista previa de "
        "la webcam, que si funciona."
    )


def main() -> int:
    """Punto de entrada del script.

    Returns:
        El codigo de salida del proceso.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Flujo completo del sistema: desbloqueo con la webcam y despues el "
            "tracker sobre clips grabados. Sin opciones muestra la webcam con el "
            "FPS incrustado, que es lo que ya funciona en este esqueleto."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Estado: el bloque de seguridad y el tracker estan por implementar; "
            "con --security o --tracker el script falla indicando que falta."
        ),
    )
    config = get_config()
    parser.add_argument(
        "--security",
        action="store_true",
        help="Ejecuta solo el bloque de desbloqueo con la webcam.",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        metavar="CLIP",
        help="Ejecuta el tracker sobre un clip grabado.",
    )
    parser.add_argument(
        "--camera", type=int, default=config.camera.index, help="Indice de la webcam."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Guarda la secuencia de transformacion en results/<modulo>/.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(
        config,
        camera=replace(config.camera, index=args.camera),
        debug=replace(config.debug, enabled=args.debug),
    )

    try:
        if args.security:
            return run_security(config)
        if args.tracker is not None:
            return run_tracker(config, args.tracker)
        return run_preview(config)
    except NotImplementedError as error:
        _logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
