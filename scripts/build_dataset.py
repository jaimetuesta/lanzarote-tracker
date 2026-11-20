"""Construye el dataset de orientacion y su division train/val/test.

La division es SIEMPRE por clips completos, nunca por fotogramas: dos
fotogramas consecutivos del mismo clip son casi identicos y repartirlos entre
particiones inflaria las metricas.

ESTADO: la construccion del dataset todavia no esta implementada. El script
arranca y falla con un mensaje que dice que falta.

Uso::

    python scripts/build_dataset.py
    python scripts/build_dataset.py --ratios 0.7 0.15 0.15 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import List

from lanzarote_tracker.capture.frames import split_by_clip, write_splits
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
            "Divide los clips en train, val y test POR CLIPS COMPLETOS y escribe "
            "la division en data/splits/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="La division nunca se hace por fotogramas sueltos.",
    )
    config = get_config()
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=list(config.classification.split_ratios),
        help="Proporciones de cada particion; deben sumar 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.classification.random_state,
        help="Semilla del reparto, para que sea reproducible.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(
        config,
        classification=replace(
            config.classification,
            split_ratios=tuple(args.ratios),
            random_state=args.seed,
        ),
    )

    clip_names: List[str] = sorted(
        path.stem for path in config.paths.clips.glob(f"*{config.capture.clip_extension}")
    )
    if not clip_names:
        _logger.error(
            "No hay clips en %s. Graba alguno con `python scripts/record_clips.py`.",
            config.paths.clips,
        )
        return 1

    try:
        splits = split_by_clip(clip_names, config.classification)
        written = write_splits(splits, config.paths)
    except (NotImplementedError, ValueError) as error:
        _logger.error("%s", error)
        return 1

    for path in written:
        _logger.info("Escrito %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
