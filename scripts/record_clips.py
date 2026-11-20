"""Graba clips del directo del aeropuerto con su JSON de metadatos.

Cada clip se guarda en `data/clips/` junto a un JSON con la marca de tiempo UTC
de inicio, que es lo que luego permite cruzarlo con OpenSky. Los clips NO se
publican: `data/clips/` esta en .gitignore.

ESTADO: la grabacion todavia no esta implementada. La extraccion de la URL del
stream (capture/stream_url.py) SI funciona, asi que con `--url-only` el script
imprime la URL directa y termina correctamente.

Uso::

    python scripts/record_clips.py --count 3 --seconds 60
    python scripts/record_clips.py --url-only
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from lanzarote_tracker.capture.recorder import record_clips
from lanzarote_tracker.capture.stream_url import get_stream_url
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
            "Graba clips del directo de Flightradar24 del aeropuerto de "
            "Lanzarote, cada uno con un JSON de metadatos con su marca de "
            "tiempo UTC de inicio."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config = get_config()
    parser.add_argument("--count", type=int, default=1, help="Numero de clips a grabar.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=config.capture.clip_seconds,
        help="Duracion de cada clip, en segundos.",
    )
    parser.add_argument(
        "--url", default=config.capture.stream_url, help="URL de la pagina del directo."
    )
    parser.add_argument(
        "--url-only",
        action="store_true",
        help="Solo extraer e imprimir la URL directa del stream, sin grabar.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(config, capture=replace(config.capture, stream_url=args.url))

    try:
        if args.url_only:
            print(get_stream_url(args.url, config.capture))
            return 0
        clips = record_clips(args.count, config.paths, config.capture, seconds=args.seconds)
    except (NotImplementedError, RuntimeError) as error:
        _logger.error("%s", error)
        return 1

    for clip in clips:
        _logger.info("%s (%s UTC)", clip.clip_path.name, clip.started_at_utc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
