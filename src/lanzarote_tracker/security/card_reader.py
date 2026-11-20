"""Lector de tarjetas por fotograma, con validacion temporal.

Une la segmentacion por color y la clasificacion de forma en una sola llamada
por fotograma, y acumula las lecturas para que una tarjeta solo se de por leida
cuando se ha reconocido IGUAL durante N fotogramas consecutivos (N =
`SecurityConfig.consecutive_frames`, 15 por defecto). Asi un reflejo o un
fotograma movido no cuentan como lectura.

La maquina de estados (:mod:`lanzarote_tracker.security.sequence_fsm`) consume
la salida de este modulo.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import SecurityConfig
from lanzarote_tracker.security.types import Card

Image = np.ndarray


def read_card(
    frame: Image,
    config: SecurityConfig,
    recorder: Optional[DebugRecorder] = None,
) -> Optional[Card]:
    """Reconoce la tarjeta presente en un unico fotograma.

    Encadena `segmentation.segment_card` (color y contorno) con
    `shape_classifier.classify_shape` (forma) y empaqueta el resultado.

    Args:
        frame: Fotograma BGR, ya corregido de distorsion si procede.
        config: Configuracion de seguridad.
        recorder: Registro opcional de la secuencia de transformacion.

    Returns:
        La tarjeta reconocida, o None si en este fotograma no hay ninguna.
    """
    raise NotImplementedError(
        "Pendiente: read_card (security/card_reader.py). "
        "segment_card + classify_shape."
    )


class StableCardReader:
    """Lector con validacion temporal: confirma una tarjeta tras N fotogramas.

    Mantiene cuantos fotogramas seguidos se ha visto la misma tarjeta y cuantos
    se lleva sin ver ninguna. La maquina de estados usa ambos contadores: el
    primero para confirmar una lectura y el segundo para exigir el hueco entre
    tarjetas.

    Attributes:
        config: Configuracion de seguridad.
    """

    def __init__(self, config: SecurityConfig) -> None:
        """Inicializa el lector.

        Args:
            config: Configuracion de seguridad, de la que se toma N
                (`consecutive_frames`).
        """
        raise NotImplementedError(
            "Pendiente: StableCardReader.__init__ (security/card_reader.py)."
        )

    def update(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Optional[Card]:
        """Procesa un fotograma y devuelve la tarjeta CONFIRMADA, si la hay.

        Devuelve la tarjeta solo en el fotograma en el que se completan los N
        fotogramas consecutivos con la misma lectura; en los demas devuelve
        None, aunque se este viendo una tarjeta.

        Args:
            frame: Fotograma BGR.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            La tarjeta confirmada en este fotograma, o None.
        """
        raise NotImplementedError(
            "Pendiente: StableCardReader.update (security/card_reader.py). "
            "read_card + contador de fotogramas consecutivos iguales."
        )

    @property
    def current_card(self) -> Optional[Card]:
        """Tarjeta que se esta viendo ahora mismo, confirmada o no."""
        raise NotImplementedError(
            "Pendiente: StableCardReader.current_card (security/card_reader.py)."
        )

    @property
    def stable_frames(self) -> int:
        """Fotogramas consecutivos con la misma lectura acumulados hasta ahora."""
        raise NotImplementedError(
            "Pendiente: StableCardReader.stable_frames (security/card_reader.py)."
        )

    @property
    def empty_frames(self) -> int:
        """Fotogramas consecutivos sin ninguna tarjeta a la vista."""
        raise NotImplementedError(
            "Pendiente: StableCardReader.empty_frames (security/card_reader.py)."
        )

    def reset(self) -> None:
        """Olvida los contadores y la tarjeta en curso."""
        raise NotImplementedError(
            "Pendiente: StableCardReader.reset (security/card_reader.py)."
        )


__all__ = ["read_card", "StableCardReader"]
