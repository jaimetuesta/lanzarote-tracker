"""Identificacion del vuelo y ficha de datos fijos.

Fuente PRIMARIA: ADS-B (`adsb.lol` / `adsb.fi` para los aviones en vivo y
`adsbdb.com` para modelo, matricula, aerolinea y ruta). Responde al momento,
que es lo que hace falta mientras se graban los clips.

Fuente SECUNDARIA: OpenSky, cuyas consultas de llegadas y salidas solo estan
disponibles al dia siguiente y sirven para confirmar a posteriori.
"""

from __future__ import annotations

from lanzarote_tracker.identification.adsb_client import (
    AdsbClient,
    RawAircraft,
    altitude_ft,
    callsign,
    icao24,
    vertical_rate_fpm,
)
from lanzarote_tracker.identification.adsbdb_client import AdsbdbClient, AircraftRoute
from lanzarote_tracker.identification.airports import (
    describe_nearest,
    format_airport,
    load_airports,
    nearest_airport,
)
from lanzarote_tracker.identification.flight_info import (
    AIRCRAFT_MODELS,
    FlightInfo,
    build_flight_info,
    infer_phase_from_route,
    resolve_model,
    resolve_route,
)
from lanzarote_tracker.identification.live_tracker import (
    FlightTracker,
    FollowedDeparture,
    TrackerState,
)
from lanzarote_tracker.identification.matcher import (
    MatchCandidate,
    RecordedCard,
    candidates_for,
    filter_by_phase,
    load_records,
    match_flight,
)
from lanzarote_tracker.identification.movement import (
    AltitudeHistory,
    distance_to_runway,
    infer_from_vertical_rate,
    infer_movement,
    is_on_runway,
)
from lanzarote_tracker.identification.opensky_client import (
    AircraftMetadata,
    FlightRecord,
    OpenSkyClient,
    StateVector,
)

__all__ = [
    "AdsbClient",
    "RawAircraft",
    "altitude_ft",
    "callsign",
    "icao24",
    "vertical_rate_fpm",
    "AdsbdbClient",
    "AircraftRoute",
    "describe_nearest",
    "format_airport",
    "load_airports",
    "nearest_airport",
    "AIRCRAFT_MODELS",
    "FlightInfo",
    "build_flight_info",
    "infer_phase_from_route",
    "resolve_model",
    "resolve_route",
    "FlightTracker",
    "FollowedDeparture",
    "TrackerState",
    "MatchCandidate",
    "RecordedCard",
    "candidates_for",
    "filter_by_phase",
    "load_records",
    "match_flight",
    "AltitudeHistory",
    "distance_to_runway",
    "infer_from_vertical_rate",
    "infer_movement",
    "is_on_runway",
    "AircraftMetadata",
    "FlightRecord",
    "OpenSkyClient",
    "StateVector",
]
