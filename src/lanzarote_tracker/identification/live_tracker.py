"""Seguimiento en vivo de los vuelos que operan en el aeropuerto.

IMPORTANTE, Y FACIL DE PASAR POR ALTO: esto hay que ejecutarlo **a la vez** que
se graban los clips. Los datos ADS-B son efimeros: dicen que aviones hay AHORA
sobre la pista, y no se pueden recuperar despues. Si se graba un clip sin tener
este seguimiento en marcha, mas adelante no habra forma de saber que avion
salia en el, salvo recurriendo a las consultas de llegadas y salidas de
OpenSky, que solo estan disponibles al dia siguiente y no siempre cuadran.

Las fichas se guardan en JSONL con su marca de tiempo UNIX, y es esa marca la
que luego permite cruzarlas con los clips grabados
(:mod:`lanzarote_tracker.identification.matcher`).

Esto NO es la ampliacion de "funcionamiento en directo": esa consiste en pasar
el PIPELINE DE VISION al directo. Aqui lo unico que va en vivo es la consulta
de datos de vuelo, que es efimera por naturaleza y tiene que ir en vivo tambien
para el sistema base, que trabaja sobre clips grabados.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from lanzarote_tracker.classification.arrival_departure import FlightPhase
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import AppConfig
from lanzarote_tracker.identification.adsb_client import (
    AdsbClient,
    RawAircraft,
    altitude_ft,
    callsign,
    icao24,
    vertical_rate_fpm,
)
from lanzarote_tracker.identification.adsbdb_client import AdsbdbClient
from lanzarote_tracker.identification.airports import describe_nearest, load_airports
from lanzarote_tracker.identification.flight_info import (
    FlightInfo,
    build_flight_info,
    infer_phase_from_route,
)
from lanzarote_tracker.identification.movement import (
    AltitudeHistory,
    infer_from_vertical_rate,
    infer_movement,
    is_on_runway,
)

_logger = get_logger(__name__)

#: Nombre del fichero JSONL donde se acumulan las fichas.
RECORDS_FILENAME = "flights.jsonl"


@dataclass
class FollowedDeparture:
    """Una salida que se sigue hasta saber a que aeropuerto ha ido.

    Attributes:
        started_at: Marca de tiempo UNIX en la que empezo el seguimiento.
        last_query: Marca de tiempo UNIX de la ultima consulta.
        last_position: Ultima posicion conocida (lat, lon, altura en pies).
    """

    started_at: int
    last_query: int
    last_position: Optional[tuple] = None


@dataclass
class TrackerState:
    """Estado del seguimiento entre iteraciones.

    Attributes:
        cards: Ultima ficha emitida por avion.
        followed: Salidas en seguimiento, por ICAO24.
        last_seen: Ultima vez que se vio cada avion, con sus datos.
    """

    cards: Dict[str, FlightInfo] = field(default_factory=dict)
    followed: Dict[str, FollowedDeparture] = field(default_factory=dict)
    last_seen: Dict[str, tuple] = field(default_factory=dict)


class FlightTracker:
    """Sigue en vivo los vuelos que operan en la pista y emite sus fichas.

    El bucle esta partido en :meth:`step`, que procesa una iteracion con la
    marca de tiempo que se le pase, de modo que se puede probar sin red y sin
    esperas reales.

    Attributes:
        config: Configuracion de la aplicacion.
        state: Estado acumulado entre iteraciones.
    """

    def __init__(
        self,
        config: AppConfig,
        adsb: Optional[AdsbClient] = None,
        adsbdb: Optional[AdsbdbClient] = None,
        airports=None,
        on_card: Optional[Callable[[FlightInfo], None]] = None,
    ) -> None:
        """Inicializa el seguidor.

        Args:
            config: Configuracion de la aplicacion.
            adsb: Cliente ADS-B. Si es None se construye uno.
            adsbdb: Cliente de adsbdb. Si es None se construye uno.
            airports: Tabla de aeropuertos. Si es None se carga del disco.
            on_card: Funcion a la que se avisa con cada ficha nueva, para
                mostrarla. Si es None, las fichas solo se guardan.
        """
        self.config: AppConfig = config
        self.identification = config.identification
        self.adsb: AdsbClient = adsb or AdsbClient(self.identification)
        self.adsbdb: AdsbdbClient = adsbdb or AdsbdbClient(self.identification)
        self.airports = load_airports(config.paths) if airports is None else airports
        self.history: AltitudeHistory = AltitudeHistory(self.identification)
        self.state: TrackerState = TrackerState()
        self.on_card: Optional[Callable[[FlightInfo], None]] = on_card

    # --- Construccion de fichas ---------------------------------------------
    def build_card(self, aircraft: RawAircraft, now: int) -> FlightInfo:
        """Monta la ficha de un avion con todo lo que se sabe de el.

        El movimiento se deduce en cascada, de lo mas fiable a lo menos:
        el historial de alturas, la velocidad vertical, la ruta habitual y, por
        ultimo, el hecho de estar parado en pista (que solo puede ser una
        salida esperando para despegar).

        Args:
            aircraft: Avion tal cual lo devuelve la API ADS-B.
            now: Marca de tiempo UNIX de la iteracion.

        Returns:
            La ficha del vuelo.
        """
        hexcode = icao24(aircraft)
        flight_callsign = callsign(aircraft)
        route = self.adsbdb.lookup(hexcode, flight_callsign)

        phase, event_time = infer_movement(self.history.get(hexcode))
        if phase is FlightPhase.UNKNOWN:
            phase = infer_from_vertical_rate(aircraft, self.identification)
        if phase is FlightPhase.UNKNOWN:
            phase = infer_phase_from_route(route, self.identification)
        if phase is FlightPhase.UNKNOWN and altitude_ft(aircraft, self.identification) == 0:
            # Parado en pista y sin haberlo visto aterrizar: espera para despegar.
            phase = FlightPhase.DEPARTURE

        return build_flight_info(
            hexcode=hexcode,
            flight_callsign=flight_callsign,
            route=route,
            phase=phase,
            config=self.identification,
            event_time=event_time,
            aircraft=aircraft,
        )

    def process(self, aircraft: RawAircraft, now: int) -> Optional[FlightInfo]:
        """Procesa un avion que esta operando en la pista.

        Args:
            aircraft: Avion tal cual lo devuelve la API ADS-B.
            now: Marca de tiempo UNIX de la iteracion.

        Returns:
            La ficha si es nueva o ha cambiado, o None si no aporta nada nuevo.
        """
        card = self.build_card(aircraft, now)
        hexcode = card.icao24
        if hexcode in self.state.followed or card == self.state.cards.get(hexcode):
            return None

        self.state.cards[hexcode] = card
        self.emit(card)

        # Solo se siguen los despegues OBSERVADOS (con hora), no los deducidos
        # de la ruta habitual: de esos no se sabe si han llegado a despegar.
        if card.phase is FlightPhase.DEPARTURE and card.event_time is not None:
            self.state.followed[hexcode] = FollowedDeparture(
                started_at=now, last_query=now
            )
        return card

    def emit(self, card: FlightInfo) -> None:
        """Guarda la ficha y avisa a quien la tenga que mostrar.

        Args:
            card: Ficha a emitir.
        """
        self.save(card)
        if self.on_card is not None:
            self.on_card(card)

    def save(self, card: FlightInfo) -> Path:
        """Anade la ficha al fichero JSONL de registros.

        Args:
            card: Ficha a guardar.

        Returns:
            La ruta del fichero escrito.
        """
        directory = Path(self.config.paths.flight_records)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / RECORDS_FILENAME
        payload = {
            "t": int(time.time()),
            "icao24": card.icao24,
            "movement": card.movement,
            "callsign": card.callsign,
            "airline": card.airline,
            "origin": card.origin,
            "destination": card.destination,
            "event_time": card.event_time,
            "aircraft_model": card.aircraft_model,
            "registration": card.registration,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    # --- Iteracion del bucle -------------------------------------------------
    def step(self, now: Optional[int] = None) -> List[FlightInfo]:
        """Ejecuta una iteracion del seguimiento.

        Args:
            now: Marca de tiempo UNIX de la iteracion. Si es None se toma del
                reloj; pasarla explicitamente permite probar sin esperas.

        Returns:
            Las fichas nuevas o actualizadas de esta iteracion.
        """
        now = int(time.time()) if now is None else now
        emitted: List[FlightInfo] = []

        aircraft_list = self.adsb.aircraft_near_airport()
        for aircraft in aircraft_list:
            altitude = altitude_ft(aircraft, self.identification)
            if altitude is None:
                continue
            self.history.record(icao24(aircraft), now, altitude)
            self.state.last_seen[icao24(aircraft)] = (now, aircraft)

        for aircraft in aircraft_list:
            if is_on_runway(aircraft, self.identification):
                card = self.process(aircraft, now)
                if card is not None:
                    emitted.append(card)

        emitted.extend(self.handle_lost(now))
        emitted.extend(self.follow_departures(now))
        return emitted

    def handle_lost(self, now: int) -> List[FlightInfo]:
        """Da por aterrizados los aviones que se pierden bajando a baja altura.

        Un avion que deja de emitir mientras desciende cerca del suelo es, casi
        siempre, uno que acaba de aterrizar: a partir de ahi ya no se le ve.

        Args:
            now: Marca de tiempo UNIX de la iteracion.

        Returns:
            Las fichas emitidas por esos aterrizajes.
        """
        emitted: List[FlightInfo] = []
        for hexcode, (seen_at, aircraft) in list(self.state.last_seen.items()):
            if now - seen_at <= self.identification.lost_after_seconds:
                continue
            altitude = altitude_ft(aircraft, self.identification)
            rate = vertical_rate_fpm(aircraft)
            if altitude is not None and 0 < altitude < self.identification.lost_below_ft and rate < 0:
                self.history.append_landing(
                    hexcode, seen_at + self.identification.poll_seconds
                )
                card = self.process(aircraft, now)
                if card is not None:
                    emitted.append(card)
            del self.state.last_seen[hexcode]
        return emitted

    def follow_departures(self, now: int) -> List[FlightInfo]:
        """Sigue a los aviones que han despegado hasta saber adonde han ido.

        Args:
            now: Marca de tiempo UNIX de la iteracion.

        Returns:
            Las fichas actualizadas con su destino.
        """
        emitted: List[FlightInfo] = []
        for hexcode, followed in list(self.state.followed.items()):
            if now - followed.started_at > self.identification.max_follow_seconds:
                del self.state.followed[hexcode]
                continue
            if now - followed.last_query < self.identification.follow_poll_seconds:
                continue
            followed.last_query = now

            position = self._resolve_landing_position(hexcode, followed)
            if position is None:
                continue

            destination = describe_nearest(
                position[0], position[1], self.airports, self.identification
            )
            del self.state.followed[hexcode]
            if destination is None:
                # Ha vuelto a tierra en el propio Lanzarote: no es un destino.
                continue

            card = self.state.cards.get(hexcode)
            if card is None:
                continue
            from dataclasses import replace

            updated = replace(card, destination=destination)
            updated = replace(
                updated, complete=updated.is_complete(self.identification)
            )
            self.state.cards[hexcode] = updated
            self.emit(updated)
            emitted.append(updated)
        return emitted

    def _resolve_landing_position(
        self, hexcode: str, followed: FollowedDeparture
    ) -> Optional[tuple]:
        """Averigua donde ha tocado tierra un avion que se estaba siguiendo.

        Args:
            hexcode: Identificador ICAO24 del avion.
            followed: Estado del seguimiento de esa salida.

        Returns:
            El par (lat, lon) donde ha aterrizado, o None si todavia vuela o no
            hay forma de saberlo.
        """
        aircraft = self.adsb.aircraft_by_hex(hexcode)
        if aircraft is not None and aircraft.get("lat") is not None:
            altitude = altitude_ft(aircraft, self.identification)
            if altitude is not None:
                followed.last_position = (
                    float(aircraft["lat"]),
                    float(aircraft["lon"]),
                    altitude,
                )
                if altitude > 0:
                    return None
                return float(aircraft["lat"]), float(aircraft["lon"])

        # Se ha perdido a baja altura: se supone que ha aterrizado cerca de su
        # ultima posicion conocida.
        last = followed.last_position
        if last is not None and last[2] < self.identification.landed_below_ft:
            return last[0], last[1]
        return None

    def run(self) -> None:
        """Ejecuta el bucle de seguimiento hasta que se interrumpa.

        Pensado para lanzarse EN PARALELO a `scripts/record_clips.py`.
        """
        _logger.info(
            "Siguiendo vuelos en %s cada %d s. Ctrl+C para parar.",
            self.identification.airport_icao,
            self.identification.poll_seconds,
        )
        try:
            while True:
                self.step()
                time.sleep(self.identification.poll_seconds)
        except KeyboardInterrupt:
            _logger.info("Seguimiento detenido por el usuario.")


__all__ = ["RECORDS_FILENAME", "FollowedDeparture", "TrackerState", "FlightTracker"]
