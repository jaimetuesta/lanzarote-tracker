"""Medida del FPS con media movil y dibujo del valor sobre el fotograma.

El video de entrega debe llevar el FPS incrustado en cada fotograma, asi que
esta utilidad es comun a todos los bucles de video del proyecto.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from lanzarote_tracker.config import VisualizationConfig

Image = np.ndarray


class FpsMeter:
    """Contador de fotogramas por segundo con media movil.

    Mide el tiempo entre llamadas consecutivas a :meth:`tick` y promedia los
    ultimos `window` intervalos, para que el valor mostrado no oscile.

    Attributes:
        window: Numero de intervalos que entran en la media movil.
    """

    def __init__(self, window: int = 30) -> None:
        """Inicializa el contador.

        Args:
            window: Numero de intervalos de la media movil. Debe ser mayor que 0.

        Raises:
            ValueError: Si `window` no es positivo.
        """
        if window <= 0:
            raise ValueError("La ventana de la media movil debe ser mayor que 0.")
        self.window: int = window
        self._intervals: Deque[float] = deque(maxlen=window)
        self._last_time: Optional[float] = None

    def tick(self) -> float:
        """Registra un fotograma nuevo y devuelve el FPS medio actual.

        Returns:
            El FPS promediado sobre la ventana, o 0.0 en el primer fotograma,
            donde todavia no hay intervalo que medir.
        """
        now = time.perf_counter()
        if self._last_time is not None:
            interval = now - self._last_time
            if interval > 0:
                self._intervals.append(interval)
        self._last_time = now
        return self.fps

    @property
    def fps(self) -> float:
        """FPS medio actual, o 0.0 si todavia no hay medidas."""
        if not self._intervals:
            return 0.0
        mean_interval = sum(self._intervals) / len(self._intervals)
        return 1.0 / mean_interval if mean_interval > 0 else 0.0

    def reset(self) -> None:
        """Vacia la ventana y olvida la ultima marca de tiempo."""
        self._intervals.clear()
        self._last_time = None


def _text_origin(
    position: str,
    frame_shape: Tuple[int, ...],
    text_size: Tuple[int, int],
    margin: int,
) -> Tuple[int, int]:
    """Calcula la esquina inferior izquierda del texto segun la posicion pedida.

    Args:
        position: "top-left", "top-right", "bottom-left" o "bottom-right".
        frame_shape: Forma del fotograma, tal cual la da `image.shape`.
        text_size: Tamano del texto (ancho, alto) devuelto por `cv2.getTextSize`.
        margin: Margen en pixeles respecto al borde.

    Returns:
        El punto (x, y) que espera `cv2.putText`.
    """
    height, width = frame_shape[0], frame_shape[1]
    text_width, text_height = text_size
    x = margin if position.endswith("left") else width - text_width - margin
    y = text_height + margin if position.startswith("top") else height - margin
    return x, y


def draw_fps(
    frame: Image,
    fps: float,
    config: Optional[VisualizationConfig] = None,
    label: str = "FPS",
) -> Image:
    """Dibuja el FPS sobre una COPIA del fotograma.

    Las funciones de dibujo de OpenCV modifican la imagen que reciben, asi que
    se copia antes para no estropear el fotograma original.

    Args:
        frame: Fotograma de entrada, que no se modifica.
        fps: Valor a mostrar.
        config: Configuracion de visualizacion (fuente, color, posicion).
        label: Etiqueta que precede al numero.

    Returns:
        Una copia del fotograma con el texto dibujado.
    """
    config = config or VisualizationConfig()
    annotated = frame.copy()
    text = f"{label}: {fps:5.1f}"
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, config.font_thickness
    )
    origin = _text_origin(
        config.fps_position, annotated.shape, (text_width, text_height), config.card_margin
    )
    # Contorno oscuro debajo para que el texto se lea sobre cualquier fondo.
    cv2.putText(
        annotated,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.font_scale,
        (0, 0, 0),
        config.font_thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.font_scale,
        config.text_color,
        config.font_thickness,
        cv2.LINE_AA,
    )
    return annotated


__all__ = ["FpsMeter", "draw_fps"]
