"""Identificacion del vuelo con OpenSky Network y ficha de datos fijos."""

from __future__ import annotations

from lanzarote_tracker.identification.flight_info import FlightInfo, build_flight_info
from lanzarote_tracker.identification.matcher import (
    MatchCandidate,
    filter_by_phase,
    match_flight,
)
from lanzarote_tracker.identification.opensky_client import (
    AircraftMetadata,
    FlightRecord,
    OpenSkyClient,
    StateVector,
)

__all__ = [
    "FlightInfo",
    "build_flight_info",
    "MatchCandidate",
    "filter_by_phase",
    "match_flight",
    "AircraftMetadata",
    "FlightRecord",
    "OpenSkyClient",
    "StateVector",
]
