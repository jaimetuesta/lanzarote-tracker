"""AMPLIACION 4 (NO BASE): seguridad reforzada con texturas.

Que hara: anadir a cada tarjeta una textura ademas de su forma y su color, de
modo que la secuencia de desbloqueo sea mas dificil de replicar: no bastaria
con ensenar un cartel del color y la forma correctos, habria que tener la
tarjeta impresa con su textura.

Idea: descriptores de textura sobre el interior del contorno ya segmentado, por
ejemplo LBP (Local Binary Patterns, disponible en
`skimage.feature.local_binary_pattern`) o filtros de Gabor, y un clasificador
sencillo sobre esos descriptores.

Fuera de las practicas: los descriptores de textura no se vieron en las
practicas de la asignatura. Justificar con bibliografia en el informe.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

Image = np.ndarray


def describe_texture(patch: Image) -> np.ndarray:
    """Calculara el descriptor de textura del interior de una tarjeta.

    Args:
        patch: Recorte del interior del contorno de la tarjeta.

    Returns:
        El vector descriptor de la textura.

    Raises:
        NotImplementedError: Siempre: es una ampliacion, no alcance base.
    """
    raise NotImplementedError(
        "AMPLIACION no implementada: seguridad con texturas "
        "(extensions/texture_security.py). No forma parte del alcance base."
    )


def classify_texture(patch: Image) -> Optional[str]:
    """Clasificara la textura de una tarjeta.

    Args:
        patch: Recorte del interior del contorno de la tarjeta.

    Returns:
        El nombre de la textura, o None si no se reconoce.

    Raises:
        NotImplementedError: Siempre: es una ampliacion.
    """
    raise NotImplementedError(
        "AMPLIACION no implementada: seguridad con texturas "
        "(extensions/texture_security.py)."
    )


__all__ = ["describe_texture", "classify_texture"]
