"""Ficha de datos del vuelo que se superpone al video.

La ficha solo lleva datos FIJOS del avion y del vuelo: indicativo, aerolinea,
modelo, matricula, origen, destino y la hora del despegue o del aterrizaje. NO
lleva altura, velocidad ni posicion, aunque las fuentes ADS-B las traigan.

El punto delicado es el origen y el destino. adsbdb devuelve la ruta HABITUAL
de un indicativo, no la del vuelo de hoy, asi que creersela sin mas llenaria la
ficha de datos falsos cuando el vuelo se sale de su patron. Por eso solo se
acepta si uno de sus extremos es Lanzarote Y ademas cuadra con el movimiento
observado; si no, el extremo desconocido se deja vacio.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from lanzarote_tracker.classification.arrival_departure import FlightPhase
from lanzarote_tracker.config import IdentificationConfig
from lanzarote_tracker.identification.adsb_client import RawAircraft
from lanzarote_tracker.identification.adsbdb_client import AircraftRoute
from lanzarote_tracker.identification.airports import format_airport

#: Modelos comerciales mas frecuentes en Lanzarote, por codigo OACI de tipo.
#: adsbdb devuelve el codigo (p. ej. "A20N") y a veces fabricante y tipo por
#: separado; esta tabla da el nombre que se lee bien en la ficha.
AIRCRAFT_MODELS: Dict[str, str] = {
    "AT45": "ATR 42-500",
    "AT46": "ATR 42-600",
    "AT72": "ATR 72",
    "AT75": "ATR 72-500",
    "AT76": "ATR 72-600",
    "A319": "Airbus A319",
    "A320": "Airbus A320",
    "A20N": "Airbus A320neo",
    "A321": "Airbus A321",
    "A21N": "Airbus A321neo",
    "B737": "Boeing 737-700",
    "B738": "Boeing 737-800",
    "B38M": "Boeing 737 MAX 8",
    "B752": "Boeing 757-200",
    "E195": "Embraer 195",
    "E295": "Embraer 195-E2",
    "CRJ9": "Bombardier CRJ900",
}


@dataclass(frozen=True)
class FlightInfo:
    """Datos fijos que se muestran en la ficha.

    Attributes:
        icao24: Identificador ICAO24 del avion. No se muestra: sirve para
            cruzar la ficha con el avion que se esta siguiendo.
        callsign: Indicativo del vuelo.
        airline: Nombre de la compania.
        registration: Matricula del avion.
        aircraft_model: Modelo del avion.
        origin: Aeropuerto de origen, como "IATA (Ciudad)".
        destination: Aeropuerto de destino.
        phase: Fase deducida, llegada o salida.
        event_time: Marca de tiempo UNIX del despegue o del aterrizaje.
        complete: True si estan todos los campos de `card_fields`.
    """

    icao24: str = ""
    callsign: str = ""
    airline: str = ""
    registration: str = ""
    aircraft_model: str = ""
    origin: str = ""
    destination: str = ""
    phase: FlightPhase = FlightPhase.UNKNOWN
    event_time: Optional[int] = None
    complete: bool = False

    @property
    def movement(self) -> str:
        """Texto en espanol de la fase: "llegada", "salida" o "pendiente"."""
        if self.phase is FlightPhase.ARRIVAL:
            return "llegada"
        if self.phase is FlightPhase.DEPARTURE:
            return "salida"
        return "pendiente"

    def event_time_label(self, config: IdentificationConfig) -> str:
        """Etiqueta de la hora, segun sea un aterrizaje o un despegue.

        Args:
            config: Configuracion de identificacion.

        Returns:
            "Hora de aterrizaje", "Hora de despegue" o la etiqueta generica.
        """
        if self.phase is FlightPhase.ARRIVAL:
            return "Hora de aterrizaje"
        if self.phase is FlightPhase.DEPARTURE:
            return "Hora de despegue"
        return config.card_labels.get("event_time", "Hora")

    def formatted_event_time(self, config: IdentificationConfig) -> str:
        """Formatea la hora del evento en UTC.

        Args:
            config: Configuracion de identificacion.

        Returns:
            La hora formateada, o `config.unknown_value` si no se conoce.
        """
        if self.event_time is None:
            return config.unknown_value
        moment = datetime.fromtimestamp(self.event_time, tz=timezone.utc)
        return moment.strftime(config.event_time_format)

    def as_lines(self, config: IdentificationConfig) -> List[Tuple[str, str]]:
        """Devuelve la ficha como pares (etiqueta, valor) listos para dibujar.

        Solo se incluyen los campos de `config.card_fields`, de modo que la
        restriccion de "solo datos fijos" vive en la configuracion y no
        repartida por el codigo de dibujo. Los campos vacios se muestran con
        `config.unknown_value`.

        Args:
            config: Configuracion de identificacion.

        Returns:
            Las lineas de la ficha, en el orden de `config.card_fields`.
        """
        lines: List[Tuple[str, str]] = []
        for name in config.card_fields:
            if name == "event_time":
                value = self.formatted_event_time(config)
                label = self.event_time_label(config)
            else:
                raw = getattr(self, name, "")
                value = str(raw) if raw else config.unknown_value
                label = config.card_labels.get(name, name)
            lines.append((label, value))
        return lines

    def is_complete(self, config: IdentificationConfig) -> bool:
        """Indica si todos los campos de la ficha tienen valor.

        Args:
            config: Configuracion de identificacion.

        Returns:
            True si ningun campo de `config.card_fields` esta vacio.
        """
        return all(
            value != config.unknown_value for _, value in self.as_lines(config)
        )


def resolve_model(
    route: AircraftRoute,
    aircraft: Optional[RawAircraft] = None,
) -> str:
    """Resuelve el nombre del modelo del avion.

    Se intenta, por orden: la tabla de modelos frecuentes, el fabricante y tipo
    que da adsbdb, y por ultimo el codigo OACI a secas, que al menos identifica
    el aparato aunque no se lea tan bien.

    Args:
        route: Datos de adsbdb.
        aircraft: Avion de la API ADS-B, que a veces trae el tipo en `t`.

    Returns:
        El modelo, o cadena vacia si no se conoce.
    """
    icao_type = route.icao_type or (aircraft or {}).get("t")
    if icao_type and icao_type in AIRCRAFT_MODELS:
        return AIRCRAFT_MODELS[icao_type]
    parts = [route.aircraft.get("manufacturer"), route.aircraft.get("type")]
    joined = " ".join(part for part in parts if part)
    return joined or (str(icao_type) if icao_type else "")


def resolve_route(
    route: AircraftRoute,
    phase: FlightPhase,
    config: IdentificationConfig,
) -> Tuple[str, str]:
    """Resuelve el origen y el destino que se van a mostrar.

    adsbdb da la ruta HABITUAL del indicativo, no la de hoy. Solo se acepta si
    cuadra con lo observado: una llegada tiene que tener Lanzarote como destino
    y una salida como origen. Si no cuadra, se descarta ENTERA, porque un vuelo
    que se sale de su patron habitual tampoco es de fiar en el otro extremo.

    Despues se rellena con Lanzarote el extremo que corresponde a la operacion
    observada, que ese si se conoce con certeza.

    Args:
        route: Datos de adsbdb.
        phase: Fase deducida.
        config: Configuracion de identificacion.

    Returns:
        El par (origen, destino) ya formateado. Un extremo desconocido queda
        como cadena vacia.
    """
    origin_record, destination_record = route.origin, route.destination
    origin_iata = origin_record.get("iata_code")
    destination_iata = destination_record.get("iata_code")

    matches = (phase is FlightPhase.ARRIVAL and destination_iata == config.airport_iata) or (
        phase is FlightPhase.DEPARTURE and origin_iata == config.airport_iata
    )
    if not matches:
        origin_record, destination_record = {}, {}

    origin = format_airport(
        origin_record.get("iata_code"), origin_record.get("municipality")
    )
    destination = format_airport(
        destination_record.get("iata_code"), destination_record.get("municipality")
    )

    if phase is FlightPhase.ARRIVAL:
        destination = destination or config.airport_label
    elif phase is FlightPhase.DEPARTURE:
        origin = origin or config.airport_label

    return origin or "", destination or ""


def infer_phase_from_route(
    route: AircraftRoute,
    config: IdentificationConfig,
) -> FlightPhase:
    """Deduce la fase a partir de la ruta habitual, como ultimo recurso.

    Solo se usa cuando no hay historial que mirar, por ejemplo porque el avion
    ya estaba en tierra cuando el sistema arranco. Si la ruta habitual pasa por
    Lanzarote, el extremo en el que aparece dice si ese vuelo llega o sale.

    Args:
        route: Datos de adsbdb.
        config: Configuracion de identificacion.

    Returns:
        La fase deducida, o UNKNOWN si la ruta no pasa por Lanzarote.
    """
    origin_iata = route.origin.get("iata_code")
    destination_iata = route.destination.get("iata_code")
    if destination_iata == config.airport_iata:
        return FlightPhase.ARRIVAL
    if origin_iata == config.airport_iata:
        return FlightPhase.DEPARTURE
    return FlightPhase.UNKNOWN


def build_flight_info(
    hexcode: str,
    flight_callsign: str,
    route: AircraftRoute,
    phase: FlightPhase,
    config: IdentificationConfig,
    event_time: Optional[int] = None,
    aircraft: Optional[RawAircraft] = None,
) -> FlightInfo:
    """Monta la ficha a partir del vuelo observado y los datos de la aeronave.

    Args:
        hexcode: Identificador ICAO24 del avion.
        flight_callsign: Indicativo del vuelo.
        route: Datos de adsbdb.
        phase: Fase deducida.
        config: Configuracion de identificacion.
        event_time: Marca de tiempo UNIX del despegue o aterrizaje.
        aircraft: Avion de la API ADS-B, para completar modelo y matricula.

    Returns:
        La ficha, con `complete` a False si falta algun campo.
    """
    origin, destination = resolve_route(route, phase, config)
    registration = route.registration or (aircraft or {}).get("r") or ""

    info = FlightInfo(
        icao24=hexcode,
        callsign=flight_callsign,
        airline=route.airline or "",
        registration=str(registration),
        aircraft_model=resolve_model(route, aircraft),
        origin=origin,
        destination=destination,
        phase=phase,
        event_time=event_time,
    )
    return replace(info, complete=info.is_complete(config))


__all__ = [
    "AIRCRAFT_MODELS",
    "FlightInfo",
    "resolve_model",
    "resolve_route",
    "infer_phase_from_route",
    "build_flight_info",
]
