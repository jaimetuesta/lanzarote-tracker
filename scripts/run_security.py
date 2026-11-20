"""Ejecuta solo el bloque de desbloqueo con la webcam.

Muestra en vivo el estado de la maquina de estados mientras se ensenan las 4
tarjetas, y termina cuando el sistema se desbloquea o se agotan los intentos.

ESTADO: el bloque de seguridad todavia no esta implementado. El script arranca
y falla con un mensaje que dice exactamente que modulos faltan.

Uso::

    python scripts/run_security.py
    python scripts/run_security.py --debug --camera 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import AppConfig, get_config
from lanzarote_tracker.security.card_reader import StableCardReader
from lanzarote_tracker.security.sequence_fsm import SequenceFsm

_logger = get_logger(__name__)


def run(config: AppConfig) -> int:
    """Ejecutara el bucle de desbloqueo sobre la webcam.

    Args:
        config: Configuracion de la aplicacion.

    Returns:
        0 si el usuario se desbloquea, 1 si no.

    Raises:
        NotImplementedError: Mientras el lector y la maquina de estados no esten
            implementados.
    """
    raise NotImplementedError(
        "El bloque de seguridad todavia no esta implementado. Faltan: "
        "security/segmentation.py, security/shape_classifier.py, "
        "security/card_reader.py y security/sequence_fsm.py."
    )
    # Asi quedara el arranque del bucle una vez implementados esos modulos:
    recorder = DebugRecorder("security", config.debug)
    reader = StableCardReader(config.security)
    fsm = SequenceFsm(config.security)


def main() -> int:
    """Punto de entrada del script.

    Returns:
        El codigo de salida del proceso.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Desbloqueo con la webcam: ensena las 4 tarjetas de la secuencia. "
            "Muestra el estado de la maquina de estados en vivo."
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

    try:
        return run(config)
    except NotImplementedError as error:
        _logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
