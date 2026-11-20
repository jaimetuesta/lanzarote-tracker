"""Deteccion, seguimiento, clasificacion y ficha sobre un clip grabado.

Es el sistema propuesto completo, sin el bloque de seguridad: se detecta el
avion cada K fotogramas, se sigue entre detecciones, se clasifica si se ve de
frente o de espaldas, se deduce si llega o sale y se superpone la ficha con los
datos fijos del vuelo. El FPS va incrustado en cada fotograma.

ESTADO: el tracker todavia no esta implementado. El script arranca y falla con
un mensaje que dice exactamente que modulos faltan.

Uso::

    python scripts/run_tracker.py data/clips/clip_20261120T173000Z.mp4
    python scripts/run_tracker.py CLIP --detector classic --tracker lk --debug
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import AppConfig, get_config
from lanzarote_tracker.detection.factory import create_detector
from lanzarote_tracker.tracking.factory import create_tracker
from lanzarote_tracker.tracking.manager import TrackingManager

_logger = get_logger(__name__)


def run(config: AppConfig, clip_path: Path, save_video: bool) -> int:
    """Ejecutara el sistema propuesto sobre un clip grabado.

    Args:
        config: Configuracion de la aplicacion.
        clip_path: Clip sobre el que trabajar.
        save_video: Si se guarda el video anotado en results/tracker/.

    Returns:
        0 si termina correctamente.

    Raises:
        NotImplementedError: Mientras el tracker no este implementado.
        FileNotFoundError: Si el clip no existe.
    """
    if not clip_path.is_file():
        raise FileNotFoundError(
            f"No existe el clip: {clip_path}. Grabalo con "
            "`python scripts/record_clips.py`."
        )
    recorder = DebugRecorder("tracker", config.debug)
    detector = create_detector(config.detection)
    tracker = create_tracker(config.tracking)
    manager = TrackingManager(detector, tracker, config.tracking)
    raise NotImplementedError(
        "El tracker todavia no esta implementado. Faltan: detection/classic.py, "
        "tracking/optical_flow.py, tracking/manager.py, classification/bow.py, "
        "identification/matcher.py y visualization/hud.py."
    )


def main() -> int:
    """Punto de entrada del script.

    Returns:
        El codigo de salida del proceso.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta deteccion, seguimiento, clasificacion e identificacion "
            "sobre un clip grabado, con la ficha y el FPS superpuestos."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Los backends 'coco' y 'cnn' estan desactivados: su uso de deep "
            "learning esta pendiente de aprobacion del profesor."
        ),
    )
    config = get_config()
    parser.add_argument("clip", type=Path, help="Clip grabado sobre el que trabajar.")
    parser.add_argument(
        "--detector",
        choices=list(config.detection.available_backends),
        default=config.detection.backend,
        help="Backend de deteccion.",
    )
    parser.add_argument(
        "--tracker",
        choices=list(config.tracking.available_backends),
        default=config.tracking.backend,
        help="Backend de seguimiento.",
    )
    parser.add_argument(
        "--every-k",
        type=int,
        default=config.tracking.detect_every_k_frames,
        help="K: cada cuantos fotogramas se vuelve a detectar.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Guarda el video anotado en results/tracker/.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Guarda la secuencia de transformacion en results/tracker/.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(
        config,
        detection=replace(config.detection, backend=args.detector),
        tracking=replace(
            config.tracking, backend=args.tracker, detect_every_k_frames=args.every_k
        ),
        debug=replace(config.debug, enabled=args.debug),
    )

    try:
        return run(config, args.clip, args.save_video)
    except (NotImplementedError, FileNotFoundError, ValueError) as error:
        _logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
