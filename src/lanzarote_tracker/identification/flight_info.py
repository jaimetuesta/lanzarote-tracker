"""Ficha de datos del vuelo que se superpone al video.

La ficha solo lleva datos FIJOS del avion y del vuelo: modelo, matricula,
origen y destino. NO lleva altura, velocidad ni posicion, aunque los vectores de
estado de OpenSky los traigan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from lanzarote_tracker.classification.arrival_departure import FlightPhase
from lanzarote_tracker.config import IdentificationConfig
from lanzarote_tracker.identification.matcher import MatchCandidate
from lanzarote_tracker.identification.opensky_client import AircraftMetadata


@dataclass(frozen=True)
class FlightInfo:
    """Datos fijos que se muestran en la ficha.

    Attributes:
        callsign: Indicativo del vuelo.
        registration: Matricula del avion.
        aircraft_model: Modelo del avion.
        origin: Aeropuerto de origen.
        destination: Aeropuerto de destino.
        phase: Fase deducida por el sistema, llegada o salida.
        complete: True si estan todos los campos. Es False mientras las
            consultas de llegadas/salidas de OpenSky no esten disponibles, cosa
            que ocurre hasta el dia siguiente del vuelo.
    """

    callsign: str = ""
    registration: str = ""
    aircraft_model: str = ""
    origin: str = ""
    destination: str = ""
    phase: FlightPhase = FlightPhase.UNKNOWN
    complete: bool = False

    def as_lines(self, config: IdentificationConfig) -> List[Tuple[str, str]]:
        """Devuelve la ficha como pares (etiqueta, valor) listos para dibujar.

        Solo se incluyen los campos de `config.card_fields`, de modo que la
        restriccion de "solo datos fijos" queda en la configuracion y no
        repartida por el codigo de dibujo.

        Args:
            config: Configuracion de identificacion.

        Returns:
            Las lineas de la ficha, en el orden de `config.card_fields`. Los
            campos vacios se muestran como "-".
        """
        raise NotImplementedError(
            "Pendiente: FlightInfo.as_lines (identification/flight_info.py)."
        )


def build_flight_info(
    candidate: Optional[MatchCandidate],
    metadata: Optional[AircraftMetadata],
    phase: FlightPhase,
) -> FlightInfo:
    """Monta la ficha a partir del vuelo emparejado y los datos de la aeronave.

    Args:
        candidate: Vuelo emparejado, o None si no se ha identificado.
        metadata: Datos fijos de la aeronave, o None si no estan disponibles.
        phase: Fase deducida por el clasificador.

    Returns:
        La ficha, con `complete` a False si falta algun campo.
    """
    raise NotImplementedError(
        "Pendiente: build_flight_info (identification/flight_info.py)."
    )


__all__ = ["FlightInfo", "build_flight_info"]
