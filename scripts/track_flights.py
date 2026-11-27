"""Sigue en vivo los vuelos del aeropuerto y registra sus fichas.

EJECUTALO A LA VEZ QUE GRABAS CLIPS. Los datos ADS-B son efimeros: dicen que
aviones hay AHORA sobre la pista y no se pueden recuperar despues. Un clip
grabado sin este seguimiento en marcha se queda sin ficha.

Las fichas se guardan en `data/flights/flights.jsonl` con su marca de tiempo
UNIX, que es lo que luego permite cruzarlas con los clips.

Uso::

    python scripts/track_flights.py
    python scripts/track_flights.py --once
    python scripts/track_flights.py --seconds 5 --quiet
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import AppConfig, get_config
from lanzarote_tracker.identification.flight_info import FlightInfo
from lanzarote_tracker.identification.live_tracker import FlightTracker

_logger = get_logger(__name__)


def make_printer(config: AppConfig, clear: bool) -> "object":
    """Construye la funcion que muestra cada ficha por pantalla.

    Args:
        config: Configuracion de la aplicacion.
        clear: Si se limpia la pantalla antes de cada ficha, para ver solo el
            vuelo actual.

    Returns:
        Una funcion que recibe una ficha y la imprime.
    """

    def show(card: FlightInfo) -> None:
        """Imprime una ficha.

        Args:
            card: Ficha a mostrar.
        """
        if clear:
            os.system("cls" if os.name == "nt" else "clear")
        for label, value in card.as_lines(config.identification):
            if value == config.identification.unknown_value:
                continue
            print(f"{label:<20}{value}")
        print()

    return show


def main() -> int:
    """Punto de entrada del script.

    Returns:
        0 si termina correctamente, 1 si falla.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sigue en vivo los vuelos que operan en la pista de Lanzarote y "
            "registra sus fichas en data/flights/. Ejecutalo A LA VEZ que "
            "`python scripts/record_clips.py`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Los datos ADS-B son efimeros: un clip grabado sin este seguimiento "
            "en marcha se queda sin ficha."
        ),
    )
    config = get_config()
    parser.add_argument(
        "--seconds",
        type=int,
        default=config.identification.poll_seconds,
        help="Segundos entre consultas.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=config.identification.search_radius_nm,
        help="Radio de busqueda alrededor del aeropuerto, en millas nauticas.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ejecutar una sola iteracion y salir. Util para comprobar que hay red.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No imprimir las fichas; solo guardarlas.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="No limpiar la pantalla entre fichas.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(
        config,
        identification=replace(
            config.identification,
            poll_seconds=args.seconds,
            search_radius_nm=args.radius,
        ),
    )

    try:
        printer = None if args.quiet else make_printer(config, clear=not args.no_clear)
        tracker = FlightTracker(config, on_card=printer)
    except FileNotFoundError as error:
        _logger.error("%s", error)
        return 1

    if args.once:
        cards = tracker.step()
        _logger.info("%d fichas en esta iteracion.", len(cards))
        if not cards:
            _logger.info(
                "Sin fichas: puede que ahora mismo no haya ningun avion operando "
                "en la pista, que es lo normal fuera de las horas punta."
            )
        return 0

    tracker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
