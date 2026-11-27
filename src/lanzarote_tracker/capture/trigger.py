"""Disparo de la grabacion cuando hay un avion operando de verdad.

POR QUE EXISTE ESTO. Grabar a ciegas no funciona. Al mirar los primeros clips
reales del directo se vio que la pista esta vacia casi todo el tiempo: una
grabacion de 16 segundos a las 16:37 no tenia ni un avion en movimiento, y
ADS-B confirmaba que no habia ninguno en 3 millas. Grabar una hora seguida son
6 GB de pista vacia y ni un ejemplo util para entrenar nada.

La solucion es que lo que ya sabemos dispare lo que queremos grabar: el
seguimiento ADS-B sabe cuando hay un avion aproximandose o despegando, asi que
se vigila y solo se graba entonces.

Detalle que importa: el stream se deja ABIERTO mientras se espera. Abrirlo
tarda varios segundos y son justo los del aterrizaje que se quiere capturar.

El disparo es por **aproximacion**, no por "ya esta en la pista": se vigila un
radio mas amplio (`trigger_radius_nm`) y una altura mas alta
(`trigger_altitude_ft`) que los de la deteccion de operaciones, para empezar a
grabar ANTES de que el avion toque tierra.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from lanzarote_tracker.capture.recorder import (
    ClipMetadata,
    open_stream,
    record_from_capture,
)
from lanzarote_tracker.capture.stream_url import get_stream_url
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import AppConfig
from lanzarote_tracker.identification.adsb_client import (
    AdsbClient,
    RawAircraft,
    altitude_ft,
    callsign,
    icao24,
)

_logger = get_logger(__name__)

#: Funcion que devuelve los aviones de la zona. Se inyecta para poder probar
#: el disparador sin red.
AircraftSource = Callable[[], List[RawAircraft]]


def is_operating(aircraft: RawAircraft, config: AppConfig) -> bool:
    """Indica si un avion esta aproximando o despegando, y no solo de paso.

    Un avion a 35.000 pies sobre el aeropuerto no es una operacion: cruza. Lo
    que interesa es el que esta bajo y cerca.

    Args:
        aircraft: Avion tal cual lo devuelve la API ADS-B.
        config: Configuracion de la aplicacion.

    Returns:
        True si merece la pena grabar por el.
    """
    altitude = altitude_ft(aircraft, config.identification)
    if altitude is None:
        return False
    return altitude <= config.capture.trigger_altitude_ft


def active_aircraft(
    aircraft_list: Sequence[RawAircraft],
    config: AppConfig,
) -> List[RawAircraft]:
    """Filtra los aviones que justifican grabar.

    Args:
        aircraft_list: Aviones de la zona.
        config: Configuracion de la aplicacion.

    Returns:
        Los que estan operando, en el mismo orden.
    """
    return [item for item in aircraft_list if is_operating(item, config)]


def describe(aircraft_list: Sequence[RawAircraft], config: AppConfig) -> str:
    """Resume los aviones que han disparado la grabacion, para el log.

    Args:
        aircraft_list: Aviones que han disparado.
        config: Configuracion de la aplicacion.

    Returns:
        Un texto del tipo "EZY5701 a 1200 ft, IBE3410 a 900 ft".
    """
    parts = []
    for aircraft in aircraft_list:
        name = callsign(aircraft) or icao24(aircraft)
        altitude = altitude_ft(aircraft, config.identification)
        parts.append(f"{name} a {altitude} ft")
    return ", ".join(parts) if parts else "ninguno"


def watch_and_record(
    config: AppConfig,
    seconds: Optional[float] = None,
    max_clips: Optional[int] = None,
    max_wait_seconds: Optional[float] = None,
    source: Optional[AircraftSource] = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> List[ClipMetadata]:
    """Vigila el trafico y graba un clip cada vez que hay una operacion.

    Mantiene el stream abierto y leyendo mientras espera, para que al disparar
    no se pierdan los primeros segundos abriendo la conexion.

    Args:
        config: Configuracion de la aplicacion.
        seconds: Duracion de cada clip. Si es None se usa
            `config.capture.clip_seconds`.
        max_clips: Numero de clips tras el cual parar. None para no parar.
        max_wait_seconds: Tiempo maximo total de vigilancia. None para no parar.
        source: Funcion que devuelve los aviones de la zona. Si es None se usa
            el cliente ADS-B real.
        sleep: Funcion de espera. Se inyecta para poder probar sin esperas.
        now: Reloj monotono. Se inyecta para poder probar sin esperas.

    Returns:
        Los metadatos de los clips grabados.

    Raises:
        RuntimeError: Si no se puede obtener la URL del stream o abrirlo.
    """
    seconds = config.capture.clip_seconds if seconds is None else seconds

    if source is None:
        client = AdsbClient(config.identification)
        radius = config.capture.trigger_radius_nm

        def source() -> List[RawAircraft]:
            """Consulta los aviones de la zona en el radio de vigilancia."""
            return client.query(
                f"lat/{config.identification.airport_lat}"
                f"/lon/{config.identification.airport_lon}"
                f"/dist/{radius}"
            )

    stream_url = get_stream_url(config.capture.stream_url, config.capture)
    capture = open_stream(stream_url)
    recorded: List[ClipMetadata] = []
    started = now()
    last_poll = -float("inf")
    cooldown_until = -float("inf")

    _logger.info(
        "Vigilando trafico en %d NM alrededor de %s. Se grabara cuando haya un "
        "avion por debajo de %d ft. Ctrl+C para parar.",
        config.capture.trigger_radius_nm,
        config.identification.airport_icao,
        config.capture.trigger_altitude_ft,
    )

    try:
        while True:
            if max_clips is not None and len(recorded) >= max_clips:
                break
            if max_wait_seconds is not None and now() - started >= max_wait_seconds:
                _logger.info("Se acaba el tiempo de vigilancia.")
                break

            moment = now()
            if moment - last_poll >= config.capture.trigger_poll_seconds:
                last_poll = moment
                active = active_aircraft(source(), config)
                if active and moment >= cooldown_until:
                    _logger.info("Operacion detectada: %s", describe(active, config))
                    recorded.append(
                        record_from_capture(
                            capture,
                            Path(config.paths.clips),
                            seconds,
                            config.capture,
                            source_url=config.capture.stream_url,
                        )
                    )
                    cooldown_until = now() + config.capture.trigger_cooldown_seconds
                    continue
                if active:
                    _logger.debug("Operacion en curso, pero en periodo de espera.")

            # Se sigue leyendo aunque no se grabe: si no, el stream se atasca y
            # al disparar habria que recuperar un bufer de varios segundos
            # viejo, justo lo que se quiere evitar.
            if not capture.grab():
                _logger.warning("El stream se ha cortado; se reabre.")
                capture.release()
                stream_url = get_stream_url(config.capture.stream_url, config.capture)
                capture = open_stream(stream_url)
            sleep(0.0)
    except KeyboardInterrupt:
        _logger.info("Vigilancia detenida por el usuario.")
    finally:
        capture.release()

    _logger.info("Grabados %d clips.", len(recorded))
    return recorded


__all__ = [
    "AircraftSource",
    "is_operating",
    "active_aircraft",
    "describe",
    "watch_and_record",
]
