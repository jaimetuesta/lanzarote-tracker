"""AMPLIACION 2 (NO BASE): compensacion del movimiento de camara con homografia.

Que hara: estimar la transformacion entre dos fotogramas consecutivos debida al
giro y al zoom de la camara del aeropuerto, y descontarla, de modo que el
movimiento que quede sea solo el del avion. Con la camara compensada se podrian
usar tecnicas que ahora no sirven, como la sustraccion de fondo.

Idea: puntos de interes en el fondo (no dentro de la caja del avion),
emparejamiento entre fotogramas y `cv2.findHomography` con RANSAC para estimar
la homografia; despues `cv2.warpPerspective` para alinear el fotograma anterior
con el actual.

Las homografias y el emparejamiento de puntos si se vieron en las practicas,
pero su uso para estabilizar video no: justificarlo con bibliografia en el
informe.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Image = np.ndarray
BoundingBox = Tuple[int, int, int, int]


def estimate_homography(
    previous_frame: Image,
    frame: Image,
    exclude: Optional[BoundingBox] = None,
) -> Optional[np.ndarray]:
    """Estimara la homografia entre dos fotogramas consecutivos.

    Args:
        previous_frame: Fotograma anterior.
        frame: Fotograma actual.
        exclude: Caja del avion, que se excluye del emparejamiento para que el
            movimiento del propio avion no contamine la estimacion del fondo.

    Returns:
        La homografia 3x3, o None si no hay emparejamientos suficientes.

    Raises:
        NotImplementedError: Siempre: es una ampliacion, no alcance base.
    """
    raise NotImplementedError(
        "AMPLIACION no implementada: compensacion de movimiento de camara "
        "(extensions/camera_motion.py). No forma parte del alcance base."
    )


def compensate(frame: Image, homography: np.ndarray) -> Image:
    """Alineara un fotograma con el anterior aplicando la homografia.

    Args:
        frame: Fotograma a alinear.
        homography: Homografia 3x3 estimada.

    Returns:
        El fotograma alineado.

    Raises:
        NotImplementedError: Siempre: es una ampliacion.
    """
    raise NotImplementedError(
        "AMPLIACION no implementada: compensacion de movimiento de camara "
        "(extensions/camera_motion.py)."
    )


__all__ = ["estimate_homography", "compensate"]
