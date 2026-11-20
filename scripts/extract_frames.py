"""Extrae fotogramas de los clips grabados, uno de cada K.

Los fotogramas se guardan en `data/frames/<nombre del clip>/`, de modo que
nunca se pierde de que clip viene cada uno: esa informacion es imprescindible
para dividir el dataset por clips completos.

ESTADO: la extraccion todavia no esta implementada. El script arranca y falla
con un mensaje que dice que falta.

Uso::

    python scripts/extract_frames.py
    python scripts/extract_frames.py --clip data/clips/clip_20261120T173000Z.mp4 --step 10
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lanzarote_tracker.capture.frames import extract_all_frames, extract_frames
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import get_config

_logger = get_logger(__name__)


def main() -> int:
    """Punto de entrada del script.

    Returns:
        0 si termina correctamente, 1 si falla o falta por implementar.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extrae fotogramas de los clips grabados, uno de cada --step, y los "
            "guarda en data/frames/<clip>/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config = get_config()
    parser.add_argument(
        "--clip",
        type=Path,
        default=None,
        help="Procesar solo este clip. Por defecto se procesan todos.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=config.capture.frame_step,
        help="Guardar uno de cada N fotogramas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.frames,
        help="Directorio raiz de salida.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(config, capture=replace(config.capture, frame_step=args.step))

    try:
        if args.clip is not None:
            written = extract_frames(args.clip, args.output, config.capture)
            _logger.info("Extraidos %d fotogramas de %s", len(written), args.clip.name)
        else:
            per_clip = extract_all_frames(config.paths, config.capture)
            total = sum(len(items) for items in per_clip.values())
            _logger.info("Extraidos %d fotogramas de %d clips", total, len(per_clip))
    except (NotImplementedError, FileNotFoundError) as error:
        _logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
