"""Ejecuta el bloque de desbloqueo con la webcam.

Muestra en vivo el estado de la maquina de estados mientras se ensenan las 4
tarjetas, y termina cuando el usuario se desbloquea o cierra la ventana.

Como se usa:

1. Ensena una tarjeta a la camara y mantenla quieta. La barra de progreso se
   llena mientras se acumulan los N fotogramas que confirman la lectura.
2. RETIRA la tarjeta antes de ensenar la siguiente: entre dos lecturas tiene
   que haber fotogramas sin tarjeta, o la segunda se ignora.
3. Repite con las cuatro. La secuencia se comprueba ENTERA al final.

Uso::

    python scripts/run_security.py
    python scripts/run_security.py --frames 10 --debug
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Optional

import cv2

from lanzarote_tracker.calibration.store import try_load_calibration
from lanzarote_tracker.calibration.undistort import Undistorter
from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.common.fps import FpsMeter, draw_fps
from lanzarote_tracker.common.io_utils import is_quit_key, open_camera
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import AppConfig, get_config
from lanzarote_tracker.security.card_reader import StableCardReader
from lanzarote_tracker.security.sequence_fsm import FrameObservation, SequenceFsm
from lanzarote_tracker.security.types import SecurityState

_logger = get_logger(__name__)

#: Color del texto segun el estado, en BGR.
STATE_COLORS = {
    SecurityState.WAITING: (255, 255, 255),
    SecurityState.READING: (0, 255, 255),
    SecurityState.CHECKING: (0, 255, 255),
    SecurityState.UNLOCKED: (0, 255, 0),
    SecurityState.ERROR: (0, 0, 255),
    SecurityState.LOCKED: (0, 0, 255),
}


def build_undistorter(config: AppConfig) -> Optional[Undistorter]:
    """Prepara la correccion de distorsion si hay calibracion guardada.

    Args:
        config: Configuracion de la aplicacion.

    Returns:
        El corrector, o None si no se va a corregir.
    """
    if not config.security.undistort:
        return None
    result = try_load_calibration(config.paths, config.calibration)
    if result is None:
        return None
    try:
        return Undistorter(result)
    except Exception as error:  # pragma: no cover - depende del hardware
        _logger.warning("No se pudo preparar la correccion de distorsion: %s", error)
        return None


def draw_overlay(
    frame,
    fsm: SequenceFsm,
    reader: StableCardReader,
    fps: float,
    config: AppConfig,
):
    """Dibuja el estado del desbloqueo sobre una COPIA del fotograma.

    Args:
        frame: Fotograma de la webcam, que NO se modifica.
        fsm: Maquina de estados.
        reader: Lector de tarjetas.
        fps: FPS actual.
        config: Configuracion de la aplicacion.

    Returns:
        Una copia del fotograma con el estado dibujado.
    """
    annotated = draw_fps(frame, fps, config.visualization)
    height, width = annotated.shape[:2]
    color = STATE_COLORS.get(fsm.state, (255, 255, 255))

    cv2.putText(
        annotated,
        fsm.status_message(),
        (12, height - 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    # Progreso de la lectura en curso: es lo que hace entendible la espera.
    bar_width = int((width - 24) * reader.progress)
    cv2.rectangle(annotated, (12, height - 30), (width - 12, height - 16), (60, 60, 60), 1)
    if bar_width > 0:
        cv2.rectangle(annotated, (12, height - 30), (12 + bar_width, height - 16), color, -1)

    # Caja de la tarjeta que se esta viendo ahora mismo.
    card = reader.current_card
    if card is not None and card.bbox is not None:
        x, y, box_width, box_height = card.bbox
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), color, 2)
        cv2.putText(
            annotated,
            str(card),
            (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def run(config: AppConfig) -> int:
    """Ejecuta el bucle de desbloqueo sobre la webcam.

    Args:
        config: Configuracion de la aplicacion.

    Returns:
        0 si el usuario se desbloquea, 1 si sale sin conseguirlo.
    """
    try:
        capture = open_camera(config.camera)
    except RuntimeError as error:
        _logger.error("%s", error)
        return 1

    undistorter = build_undistorter(config)
    if undistorter is None and config.security.undistort:
        _logger.warning(
            "Se continua SIN corregir la distorsion. No impide desbloquear, "
            "pero las figuras cerca del borde salen algo curvadas."
        )

    recorder = DebugRecorder("security", config.debug)
    reader = StableCardReader(config.security)
    fsm = SequenceFsm(config.security)
    meter = FpsMeter(window=config.visualization.fps_window)

    window_name = "Desbloqueo - 'q' o ESC para salir"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.camera.width, config.camera.height)
    _logger.info(
        "Ensena las %d tarjetas de la secuencia, RETIRANDOLAS entre una y otra.",
        config.security.sequence_length,
    )

    previous_state = fsm.state
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                _logger.warning("Fotograma perdido; se cierra la camara.")
                break
            if config.camera.mirror:
                frame = cv2.flip(frame, 1)
            if undistorter is not None:
                frame = undistorter(frame)

            confirmed = reader.update(frame, recorder=recorder)
            if confirmed is not None:
                _logger.info("Tarjeta leida: %s", confirmed)
                # Solo se guarda la secuencia de transformacion de una lectura
                # buena: es la figura que pide el informe, no una por fotograma.
                recorder.save_all(prefix=f"{fsm.current_index}_")
            recorder.clear()

            state = fsm.update(
                FrameObservation(
                    timestamp=time.monotonic(),
                    card_visible=reader.current_card is not None,
                    confirmed_card=confirmed,
                )
            )
            if state is not previous_state:
                _logger.info("%s", fsm.status_message())
                previous_state = state

            cv2.imshow(
                window_name, draw_overlay(frame, fsm, reader, meter.tick(), config)
            )

            if fsm.is_unlocked:
                cv2.waitKey(1500)
                break
            if is_quit_key(cv2.waitKey(1) & 0xFF, config.camera.quit_keys):
                break
    except KeyboardInterrupt:
        _logger.info("Interrumpido por el usuario.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if fsm.is_unlocked:
        _logger.info("Desbloqueado.")
        return 0
    _logger.info("Se ha salido sin desbloquear.")
    return 1


def main() -> int:
    """Punto de entrada del script.

    Returns:
        El codigo de salida del proceso.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Desbloqueo con la webcam: ensena las 4 tarjetas de la secuencia, "
            "retirandolas entre una y otra. Muestra el estado en vivo."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config = get_config()
    parser.add_argument(
        "--camera", type=int, default=config.camera.index, help="Indice de la webcam."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=config.security.consecutive_frames,
        help="N: fotogramas consecutivos para dar una tarjeta por leida.",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="No aplicar la correccion de distorsion de la calibracion.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Guarda la secuencia de transformacion en results/security/.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(
        config,
        camera=replace(config.camera, index=args.camera),
        security=replace(
            config.security,
            consecutive_frames=args.frames,
            undistort=not args.no_undistort,
        ),
        debug=replace(config.debug, enabled=args.debug),
    )
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
