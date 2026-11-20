"""Emparejamiento entre el avion de la imagen y un vuelo de OpenSky.

El sistema no sabe que avion esta mirando: sabe que hay un avion en la imagen,
en que instante UTC se grabo ese fotograma (de los metadatos del clip) y si
aterriza o despega (de la clasificacion). Con eso se busca en OpenSky el vuelo
compatible.

Criterios de emparejamiento, en orden:

1. Ventana temporal alrededor de la marca de tiempo del fotograma
   (`IdentificationConfig.query_window_hours`).
2. Aviones dentro del recuadro geografico del aeropuerto.
3. Coherencia con la fase deducida: una llegada se busca entre los vuelos que
   aterrizan y una salida entre los que despegan.

Fuera de las practicas: el emparejamiento no es una tecnica de vision vista en
clase, es logica de datos; en el informe se describe como parte del sistema
propuesto y se discuten sus casos ambiguos (dos aviones a la vez en el recuadro).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

from lanzarote_tracker.classification.arrival_departure import FlightPhase
from lanzarote_tracker.config import IdentificationConfig
from lanzarote_tracker.identification.opensky_client import (
    FlightRecord,
    OpenSkyClient,
    StateVector,
)


@dataclass(frozen=True)
class MatchCandidate:
    """Un vuelo candidato a ser el avion que se ve en la imagen.

    Attributes:
        state: Vector de estado del avion.
        flight: Registro de llegada o salida, si se ha encontrado. Es None
            mientras las consultas de llegadas/salidas no esten disponibles,
            cosa que ocurre hasta el dia siguiente.
        score: Puntuacion del candidato; a mayor, mejor encaje.
    """

    state: StateVector
    flight: Optional[FlightRecord] = None
    score: float = 0.0


def filter_by_phase(
    candidates: Sequence[MatchCandidate],
    phase: FlightPhase,
) -> List[MatchCandidate]:
    """Descarta los candidatos incompatibles con la fase deducida.

    Args:
        candidates: Candidatos a filtrar.
        phase: Fase deducida por el clasificador.

    Returns:
        Los candidatos compatibles. Si la fase es UNKNOWN se devuelven todos.
    """
    raise NotImplementedError(
        "Pendiente: filter_by_phase (identification/matcher.py). "
        "Una llegada solo casa con vuelos cuyo arrival_airport sea el aeropuerto."
    )


def match_flight(
    client: OpenSkyClient,
    frame_time_utc: datetime,
    phase: FlightPhase,
    config: IdentificationConfig,
) -> Optional[MatchCandidate]:
    """Busca el vuelo que mejor encaja con lo que se ve en el fotograma.

    Args:
        client: Cliente de OpenSky ya construido.
        frame_time_utc: Marca de tiempo UTC del fotograma, calculada a partir
            de la del clip mas el desplazamiento dentro del video.
        phase: Fase deducida (llegada, salida o desconocida).
        config: Configuracion de identificacion.

    Returns:
        El mejor candidato, o None si no hay ninguno compatible.

    Raises:
        RuntimeError: Si las consultas a OpenSky fallan.
    """
    raise NotImplementedError(
        "Pendiente: match_flight (identification/matcher.py). "
        "get_states_in_bbox + filter_by_phase + puntuacion por cercania temporal."
    )


__all__ = ["MatchCandidate", "filter_by_phase", "match_flight"]
