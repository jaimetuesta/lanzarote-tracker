"""Deduccion de llegada o salida a partir de la orientacion y la pista.

La orientacion por si sola no basta: ver un avion de frente puede significar
que esta aterrizando o que esta despegando hacia la camara, segun por que
cabecera se este usando la pista. Por eso se combina la orientacion
(frente/espaldas) con el sentido de uso de la pista
(`ClassificationConfig.runway_in_use`, cabeceras 03 y 21 en GCRR).

La decision se estabiliza en el tiempo: no se da un veredicto hasta tener
`config.decision_frames` fotogramas coherentes, para que un fotograma mal
clasificado no cambie la ficha que se muestra en pantalla.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from lanzarote_tracker.classification.base import Orientation
from lanzarote_tracker.config import ClassificationConfig


class FlightPhase(str, Enum):
    """Fase del vuelo deducida.

    Attributes:
        ARRIVAL: El avion esta aterrizando.
        DEPARTURE: El avion esta despegando.
        UNKNOWN: Todavia no hay fotogramas coherentes suficientes.
    """

    ARRIVAL = "arrival"
    DEPARTURE = "departure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PhaseDecision:
    """Veredicto sobre la fase del vuelo.

    Attributes:
        phase: Fase deducida.
        confidence: Proporcion de fotogramas coherentes con la decision.
        frames: Fotogramas acumulados en la decision.
    """

    phase: FlightPhase
    confidence: float = 0.0
    frames: int = 0


def infer_phase(orientation: Orientation, runway: str) -> FlightPhase:
    """Deduce la fase del vuelo de un solo fotograma.

    Combina la orientacion con el sentido de uso de la pista. Con la camara
    situada como esta en el directo de Flightradar24:

    - cabecera en uso 03: un avion de frente llega y uno de espaldas sale;
    - cabecera en uso 21: al reves.

    TODO: confirmar esta tabla observando clips reales de las dos cabeceras
    antes de dar por buena la correspondencia, y documentarla en el informe con
    una figura.

    Args:
        orientation: Orientacion clasificada en el fotograma.
        runway: Cabecera en uso, "03" o "21".

    Returns:
        La fase deducida para ese fotograma.

    Raises:
        ValueError: Si la cabecera no es una de las configuradas.
    """
    raise NotImplementedError(
        "Pendiente: infer_phase (classification/arrival_departure.py). "
        "Tabla orientacion x cabecera en uso. "
        "Fuera de las practicas: justificar el criterio en el informe."
    )


class PhaseDecider:
    """Acumula veredictos por fotograma hasta dar una decision estable.

    Attributes:
        config: Configuracion de clasificacion, de la que se toman la cabecera
            en uso y `decision_frames`.
    """

    def __init__(self, config: ClassificationConfig) -> None:
        """Inicializa el decisor sin veredicto.

        Args:
            config: Configuracion de clasificacion.
        """
        raise NotImplementedError(
            "Pendiente: PhaseDecider.__init__ (classification/arrival_departure.py)."
        )

    def update(self, orientation: Optional[Orientation]) -> PhaseDecision:
        """Anade la orientacion de un fotograma y devuelve la decision actual.

        Args:
            orientation: Orientacion del fotograma, o None si en ese fotograma
                no habia avion que clasificar.

        Returns:
            La decision acumulada, que sera UNKNOWN hasta reunir
            `config.decision_frames` fotogramas coherentes.
        """
        raise NotImplementedError(
            "Pendiente: PhaseDecider.update (classification/arrival_departure.py)."
        )

    def reset(self) -> None:
        """Olvida los fotogramas acumulados."""
        raise NotImplementedError(
            "Pendiente: PhaseDecider.reset (classification/arrival_departure.py)."
        )


__all__ = ["FlightPhase", "PhaseDecision", "infer_phase", "PhaseDecider"]
