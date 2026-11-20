"""Ficha de datos superpuesta al video.

Dibuja un panel semitransparente con los datos fijos del vuelo: modelo,
matricula, origen y destino. Sin altura, sin velocidad y sin posicion.

El panel se compone con cv2.addWeighted sobre una copia del fotograma, que es
la forma habitual de conseguir transparencia en OpenCV: se dibuja el
rectangulo opaco sobre una capa aparte y se mezcla con el fotograma usando
`VisualizationConfig.card_alpha`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from lanzarote_tracker.config import IdentificationConfig, VisualizationConfig
from lanzarote_tracker.identification.flight_info import FlightInfo

Image = np.ndarray


def draw_info_card(
    frame: Image,
    info: FlightInfo,
    visualization: Optional[VisualizationConfig] = None,
    identification: Optional[IdentificationConfig] = None,
    anchor: str = "top-right",
) -> Image:
    """Superpone la ficha de datos sobre una COPIA del fotograma.

    Args:
        frame: Fotograma de entrada, que NO se modifica.
        info: Datos del vuelo ya resueltos.
        visualization: Configuracion de visualizacion (tamano, colores, alpha).
        identification: Configuracion de identificacion, de la que se toman los
            campos que se muestran (`card_fields`).
        anchor: Esquina donde se ancla la ficha.

    Returns:
        Una copia del fotograma con la ficha dibujada.
    """
    raise NotImplementedError(
        "Pendiente: draw_info_card (visualization/info_card.py). "
        "frame.copy() + cv2.rectangle sobre una capa + cv2.addWeighted + "
        "cv2.putText por cada linea de info.as_lines()."
    )


def card_size(info: FlightInfo, config: VisualizationConfig) -> Tuple[int, int]:
    """Calcula el tamano en pixeles que ocupara la ficha.

    Args:
        info: Datos del vuelo.
        config: Configuracion de visualizacion.

    Returns:
        El par (ancho, alto) de la ficha.
    """
    raise NotImplementedError(
        "Pendiente: card_size (visualization/info_card.py)."
    )


__all__ = ["draw_info_card", "card_size"]
