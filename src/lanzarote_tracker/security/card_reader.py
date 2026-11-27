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

import cv2
import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import SecurityConfig
from lanzarote_tracker.security.segmentation import segment_card
from lanzarote_tracker.security.shape_classifier import (
    approximate_polygon,
    circularity,
    classify_shape,
)
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
        La tarjeta reconocida, o None si en este fotograma no hay ninguna o si
        su forma no encaja en las cuatro admitidas.
    """
    found = segment_card(frame, config, recorder=recorder)
    if found is None:
        return None

    color, contour = found
    shape = classify_shape(contour, config)
    if shape is None:
        return None

    x, y, width, height = cv2.boundingRect(contour)
    return Card(
        shape=shape,
        color=color,
        bbox=(int(x), int(y), int(width), int(height)),
        area=float(cv2.contourArea(contour)),
        circularity=circularity(contour),
        vertices=len(approximate_polygon(contour, config)),
    )


class StableCardReader:
    """Lector con validacion temporal: confirma una tarjeta tras N fotogramas.

    Mantiene cuantos fotogramas seguidos se ha visto la misma tarjeta y cuantos
    se lleva sin ver ninguna. La maquina de estados usa ambos contadores: el
    primero para confirmar una lectura y el segundo para exigir el hueco entre
    tarjetas.

    Lo que se compara entre fotogramas es `Card.key`, es decir, forma y color.
    El area y la caja cambian en cada fotograma aunque la tarjeta este quieta,
    asi que compararlas enteras no confirmaria nunca.

    Attributes:
        config: Configuracion de seguridad.
    """

    def __init__(self, config: SecurityConfig) -> None:
        """Inicializa el lector.

        Args:
            config: Configuracion de seguridad, de la que se toma N
                (`consecutive_frames`).
        """
        self.config: SecurityConfig = config
        self._current: Optional[Card] = None
        self._stable_frames: int = 0
        self._empty_frames: int = 0

    def update(
        self,
        frame: Image,
        recorder: Optional[DebugRecorder] = None,
    ) -> Optional[Card]:
        """Procesa un fotograma y devuelve la tarjeta CONFIRMADA, si la hay.

        Devuelve la tarjeta solo en el fotograma en el que se completan los N
        fotogramas consecutivos con la misma lectura; en los demas devuelve
        None, aunque se este viendo una tarjeta. Si la tarjeta se mantiene
        delante de la camara, el contador sigue subiendo pero NO se vuelve a
        confirmar: una tarjeta quieta es una lectura, no muchas.

        Args:
            frame: Fotograma BGR.
            recorder: Registro opcional de la secuencia de transformacion.

        Returns:
            La tarjeta confirmada en este fotograma, o None.
        """
        card = read_card(frame, self.config, recorder=recorder)

        if card is None:
            self._current = None
            self._stable_frames = 0
            self._empty_frames += 1
            return None

        self._empty_frames = 0
        if self._current is not None and card.key == self._current.key:
            self._stable_frames += 1
        else:
            self._stable_frames = 1
        self._current = card

        if self._stable_frames == self.config.consecutive_frames:
            return card
        return None

    @property
    def current_card(self) -> Optional[Card]:
        """Tarjeta que se esta viendo ahora mismo, confirmada o no."""
        return self._current

    @property
    def stable_frames(self) -> int:
        """Fotogramas consecutivos con la misma lectura acumulados hasta ahora."""
        return self._stable_frames

    @property
    def empty_frames(self) -> int:
        """Fotogramas consecutivos sin ninguna tarjeta a la vista."""
        return self._empty_frames

    @property
    def progress(self) -> float:
        """Fraccion de los N fotogramas ya acumulada, entre 0 y 1.

        Sirve para dibujar una barra de progreso mientras se sostiene la
        tarjeta, que es lo que hace entendible la espera.
        """
        if self.config.consecutive_frames <= 0:
            return 1.0
        return min(1.0, self._stable_frames / float(self.config.consecutive_frames))

    def reset(self) -> None:
        """Olvida los contadores y la tarjeta en curso."""
        self._current = None
        self._stable_frames = 0
        self._empty_frames = 0


__all__ = ["read_card", "StableCardReader"]
