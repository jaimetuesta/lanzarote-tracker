"""Emparejamiento entre el avion de un clip y una ficha registrada.

Es la costura entre la parte de vision y la de datos. El sistema no sabe que
avion esta mirando: sabe que hay un avion en la imagen, en que instante UTC se
grabo ese fotograma (de los metadatos del clip) y si aterriza o despega (de la
clasificacion por vision). Con eso se busca entre las fichas que
:mod:`lanzarote_tracker.identification.live_tracker` fue registrando MIENTRAS
se grababa.

Criterios, en orden:

1. **Ventana temporal** alrededor de la marca de tiempo del fotograma.
2. **Coherencia con la fase** deducida por vision: una llegada solo casa con
   fichas de llegada.
3. A igualdad de lo anterior, la ficha mas cercana en el tiempo.

El resultado tiene un segundo uso, tan importante como el primero: la fase de
la ficha procede de los datos ADS-B, no de la imagen, asi que sirve de
**etiqueta de referencia** para entrenar y medir el clasificador de vision. Ver
el aviso de :mod:`lanzarote_tracker.identification.movement` sobre no mezclar
las dos fuentes al evaluar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from lanzarote_tracker.classification.arrival_departure import FlightPhase
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import IdentificationConfig, PathsConfig
from lanzarote_tracker.identification.flight_info import FlightInfo
from lanzarote_tracker.identification.live_tracker import RECORDS_FILENAME

_logger = get_logger(__name__)

#: Texto a fase, para releer las fichas guardadas.
_PHASE_BY_NAME = {
    "llegada": FlightPhase.ARRIVAL,
    "salida": FlightPhase.DEPARTURE,
    "pendiente": FlightPhase.UNKNOWN,
}


@dataclass(frozen=True)
class RecordedCard:
    """Una ficha tal como quedo guardada, con el momento en que se registro.

    Attributes:
        recorded_at: Marca de tiempo UNIX en la que se escribio la ficha.
        info: La ficha.
    """

    recorded_at: int
    info: FlightInfo


@dataclass(frozen=True)
class MatchCandidate:
    """Una ficha candidata a corresponder con el avion del clip.

    Attributes:
        card: La ficha registrada.
        seconds_apart: Distancia en segundos entre el fotograma y la ficha.
        phase_matches: Si su fase coincide con la deducida por vision.
    """

    card: RecordedCard
    seconds_apart: float
    phase_matches: bool

    @property
    def info(self) -> FlightInfo:
        """La ficha del candidato."""
        return self.card.info


def load_records(paths: Optional[PathsConfig] = None) -> List[RecordedCard]:
    """Lee las fichas registradas por el seguimiento en vivo.

    Args:
        paths: Rutas del proyecto.

    Returns:
        Las fichas, ordenadas por el momento en que se registraron. Lista vacia
        si todavia no se ha registrado ninguna.
    """
    paths = paths or PathsConfig()
    path = Path(paths.flight_records) / RECORDS_FILENAME
    if not path.is_file():
        _logger.warning(
            "No hay fichas registradas en %s. Tienen que grabarse A LA VEZ que "
            "los clips, con `python scripts/track_flights.py`.",
            path,
        )
        return []

    records: List[RecordedCard] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _logger.warning("Linea %d ilegible en %s; se omite.", number, path)
            continue
        records.append(
            RecordedCard(
                recorded_at=int(payload.get("t", 0)),
                info=FlightInfo(
                    icao24=str(payload.get("icao24") or ""),
                    callsign=str(payload.get("callsign") or ""),
                    airline=str(payload.get("airline") or ""),
                    registration=str(payload.get("registration") or ""),
                    aircraft_model=str(payload.get("aircraft_model") or ""),
                    origin=str(payload.get("origin") or ""),
                    destination=str(payload.get("destination") or ""),
                    phase=_PHASE_BY_NAME.get(
                        str(payload.get("movement") or ""), FlightPhase.UNKNOWN
                    ),
                    event_time=payload.get("event_time"),
                ),
            )
        )
    records.sort(key=lambda record: record.recorded_at)
    return records


def candidates_for(
    records: Sequence[RecordedCard],
    frame_time_utc: datetime,
    phase: FlightPhase,
    config: IdentificationConfig,
    window_seconds: float = 300.0,
) -> List[MatchCandidate]:
    """Selecciona las fichas compatibles con un fotograma.

    Args:
        records: Fichas registradas.
        frame_time_utc: Marca de tiempo UTC del fotograma, calculada a partir
            de la del clip mas el desplazamiento dentro del video.
        phase: Fase deducida por vision. Con UNKNOWN no se filtra por fase.
        config: Configuracion de identificacion.
        window_seconds: Mitad de la ventana temporal de busqueda.

    Returns:
        Los candidatos dentro de la ventana, ordenados por cercania temporal y
        poniendo delante los que ademas coinciden en fase.
    """
    if frame_time_utc.tzinfo is None:
        frame_time_utc = frame_time_utc.replace(tzinfo=timezone.utc)
    target = frame_time_utc.timestamp()

    found: List[MatchCandidate] = []
    for record in records:
        distance = abs(record.recorded_at - target)
        if distance > window_seconds:
            continue
        found.append(
            MatchCandidate(
                card=record,
                seconds_apart=distance,
                phase_matches=(
                    phase is FlightPhase.UNKNOWN or record.info.phase is phase
                ),
            )
        )
    # Primero los que coinciden en fase y, dentro de cada grupo, los mas cercanos.
    found.sort(key=lambda candidate: (not candidate.phase_matches, candidate.seconds_apart))
    return found


def match_flight(
    frame_time_utc: datetime,
    phase: FlightPhase,
    config: IdentificationConfig,
    paths: Optional[PathsConfig] = None,
    records: Optional[Sequence[RecordedCard]] = None,
    window_seconds: float = 300.0,
) -> Optional[MatchCandidate]:
    """Busca la ficha que mejor encaja con lo que se ve en el fotograma.

    Args:
        frame_time_utc: Marca de tiempo UTC del fotograma.
        phase: Fase deducida por vision.
        config: Configuracion de identificacion.
        paths: Rutas del proyecto, para leer las fichas si no se pasan.
        records: Fichas ya cargadas, para no releer el fichero en un bucle.
        window_seconds: Mitad de la ventana temporal de busqueda.

    Returns:
        El mejor candidato, o None si no hay ninguno compatible. Que no haya
        ninguno suele significar que el seguimiento en vivo no estaba en marcha
        cuando se grabo el clip.
    """
    available = load_records(paths) if records is None else list(records)
    found = candidates_for(available, frame_time_utc, phase, config, window_seconds)
    if not found:
        return None
    best = found[0]
    if not best.phase_matches:
        _logger.info(
            "La ficha mas cercana (%s) no coincide con la fase deducida por "
            "vision (%s). Puede ser un fallo del clasificador o que hubiera dos "
            "aviones a la vez.",
            best.info.movement,
            phase.value,
        )
    return best


def filter_by_phase(
    candidates: Sequence[MatchCandidate],
    phase: FlightPhase,
) -> List[MatchCandidate]:
    """Descarta los candidatos incompatibles con la fase deducida.

    Args:
        candidates: Candidatos a filtrar.
        phase: Fase deducida por vision.

    Returns:
        Los candidatos compatibles. Si la fase es UNKNOWN se devuelven todos.
    """
    if phase is FlightPhase.UNKNOWN:
        return list(candidates)
    return [candidate for candidate in candidates if candidate.info.phase is phase]


__all__ = [
    "RecordedCard",
    "MatchCandidate",
    "load_records",
    "candidates_for",
    "match_flight",
    "filter_by_phase",
]
