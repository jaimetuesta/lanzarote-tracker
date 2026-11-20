"""Capa de informacion superpuesta al video: FPS, estado y ficha.

Reune en una sola llamada todo lo que se pinta sobre cada fotograma, de modo
que los scripts no tengan que acordarse del orden ni de copiar la imagen.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from lanzarote_tracker.config import IdentificationConfig, VisualizationConfig
from lanzarote_tracker.identification.flight_info import FlightInfo

Image = np.ndarray
BoundingBox = Tuple[int, int, int, int]


def draw_hud(
    frame: Image,
    fps: float,
    bbox: Optional[BoundingBox] = None,
    label: str = "",
    info: Optional[FlightInfo] = None,
    visualization: Optional[VisualizationConfig] = None,
    identification: Optional[IdentificationConfig] = None,
) -> Image:
    """Dibuja caja, etiqueta, ficha y FPS sobre una COPIA del fotograma.

    El FPS se dibuja SIEMPRE que `visualization.draw_fps` sea True, porque el
    video de entrega debe llevarlo incrustado.

    Args:
        frame: Fotograma de entrada, que NO se modifica.
        fps: FPS actual, de
            :class:`~lanzarote_tracker.common.fps.FpsMeter`.
        bbox: Caja del avion, si hay.
        label: Texto de la caja, p. ej. "avion (seguimiento)".
        info: Ficha del vuelo, si ya se ha identificado.
        visualization: Configuracion de visualizacion.
        identification: Configuracion de identificacion.

    Returns:
        Una copia del fotograma con toda la capa de informacion dibujada.
    """
    raise NotImplementedError(
        "Pendiente: draw_hud (visualization/hud.py). "
        "draw_box + draw_info_card + draw_fps, en ese orden."
    )


def draw_security_hud(
    frame: Image,
    fps: float,
    status_message: str,
    cards_read: int,
    sequence_length: int,
    visualization: Optional[VisualizationConfig] = None,
) -> Image:
    """Dibuja el estado del desbloqueo sobre una COPIA del fotograma.

    Args:
        frame: Fotograma de la webcam, que NO se modifica.
        fps: FPS actual.
        status_message: Mensaje de
            :meth:`~lanzarote_tracker.security.sequence_fsm.SequenceFsm.status_message`.
        cards_read: Tarjetas ya confirmadas en el intento en curso.
        sequence_length: Longitud total de la secuencia.
        visualization: Configuracion de visualizacion.

    Returns:
        Una copia del fotograma con el estado y el FPS dibujados.
    """
    raise NotImplementedError(
        "Pendiente: draw_security_hud (visualization/hud.py). "
        "draw_text del estado + indicador de progreso + draw_fps."
    )


__all__ = ["draw_hud", "draw_security_hud"]
