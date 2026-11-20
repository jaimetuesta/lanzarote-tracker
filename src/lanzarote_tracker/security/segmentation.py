"""Segmentacion de las tarjetas por color.

Pipeline por fotograma, en este orden:

1. Correccion de distorsion con la calibracion de la webcam (Practica 2).
2. Conversion a HSV y umbral por rango de tono (Practica 1: cv2.cvtColor +
   cv2.inRange). El ROJO necesita DOS rangos unidos con cv2.bitwise_or porque
   su tono esta partido por el extremo del circulo cromatico; en OpenCV H va de
   0 a 179.
3. Apertura y cierre morfologicos para quitar ruido y cerrar huecos
   (Practica 3: cv2.morphologyEx con MORPH_OPEN y MORPH_CLOSE).
4. Contorno de mayor area que supere el area minima (Practica 4:
   cv2.findContours + cv2.contourArea).

Cada paso se entrega al :class:`~lanzarote_tracker.common.debug.DebugRecorder`
para la figura de la secuencia de transformacion del informe.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import HsvRange, SecurityConfig
from lanzarote_tracker.security.types import Color

Image = np.ndarray
Mask = np.ndarray
Contour = np.ndarray


def build_color_mask(
    hsv: Image,
    ranges: Tuple[HsvRange, ...],
) -> Mask:
    """Construye la mascara binaria de un color a partir de sus rangos HSV.

    Practica 1: cv2.inRange por cada rango y cv2.bitwise_or para unirlos. El
    rojo pasa dos rangos (tono cerca de 0 y tono cerca de 179) y el resto de
    colores uno solo.

    Args:
        hsv: Imagen ya convertida a HSV.
        ranges: Rangos del color, cada uno como ((h,s,v) minimo, (h,s,v) maximo),
            con H entre 0 y 179.

    Returns:
        Mascara binaria de 8 bits con 255 en los pixeles del color.
    """
    raise NotImplementedError(
        "Pendiente: build_color_mask (security/segmentation.py). "
        "Practica 1: cv2.inRange por rango + cv2.bitwise_or."
    )


def clean_mask(mask: Mask, config: SecurityConfig) -> Mask:
    """Limpia una mascara con apertura y cierre morfologicos.

    Practica 3: cv2.morphologyEx con MORPH_OPEN (quita el ruido de sal) y
    despues MORPH_CLOSE (cierra los huecos dentro de la tarjeta), con el
    elemento estructurante y el numero de iteraciones de la configuracion.

    Args:
        mask: Mascara binaria de entrada, que no se modifica.
        config: Configuracion de seguridad, de la que se toman el tamano del
            kernel y las iteraciones.

    Returns:
        Una mascara nueva, limpia.
    """
    raise NotImplementedError(
        "Pendiente: clean_mask (security/segmentation.py). "
        "Practica 3: cv2.morphologyEx MORPH_OPEN + MORPH_CLOSE."
    )


def largest_contour(mask: Mask, min_area: float) -> Optional[Contour]:
    """Devuelve el contorno de mayor area de una mascara, si supera el minimo.

    Practica 4: cv2.findContours en modo RETR_EXTERNAL con
    CHAIN_APPROX_SIMPLE, y seleccion por cv2.contourArea.

    Args:
        mask: Mascara binaria.
        min_area: Area minima en pixeles para aceptar el contorno. Descarta
            reflejos y objetos pequenos del fondo del mismo color.

    Returns:
        El contorno mayor, o None si no hay ninguno que supere `min_area`.
    """
    raise NotImplementedError(
        "Pendiente: largest_contour (security/segmentation.py). "
        "Practica 4: cv2.findContours + max por cv2.contourArea."
    )


def segment_card(
    frame: Image,
    config: SecurityConfig,
    recorder: Optional[DebugRecorder] = None,
) -> Optional[Tuple[Color, Contour]]:
    """Segmenta la tarjeta presente en el fotograma y devuelve su color y contorno.

    Prueba los cuatro colores de `config.hsv_ranges` y se queda con el que da el
    contorno de mayor area, de forma que un fondo del mismo tono pero pequeno no
    gane a la tarjeta.

    Args:
        frame: Fotograma BGR, ya corregido de distorsion si procede.
        config: Configuracion de seguridad.
        recorder: Registro opcional de la secuencia de transformacion, al que se
            entregan el fotograma original, la imagen HSV, cada mascara de color
            y la mascara limpia ganadora.

    Returns:
        El par (color, contorno) de la tarjeta, o None si no hay ninguna tarjeta
        valida en el fotograma.
    """
    raise NotImplementedError(
        "Pendiente: segment_card (security/segmentation.py). "
        "Practicas 1, 3 y 4: HSV + inRange + morfologia + findContours."
    )


def color_masks(
    frame: Image,
    config: SecurityConfig,
    recorder: Optional[DebugRecorder] = None,
) -> Dict[Color, Mask]:
    """Calcula la mascara limpia de cada color del fotograma.

    Util para `scripts/tune_hsv.py` y para las figuras del informe, donde
    interesa ver las cuatro mascaras a la vez.

    Args:
        frame: Fotograma BGR.
        config: Configuracion de seguridad.
        recorder: Registro opcional de la secuencia de transformacion.

    Returns:
        Un diccionario de color a mascara binaria ya limpia.
    """
    raise NotImplementedError(
        "Pendiente: color_masks (security/segmentation.py). "
        "build_color_mask + clean_mask para cada color de config.hsv_ranges."
    )


__all__ = [
    "Mask",
    "Contour",
    "build_color_mask",
    "clean_mask",
    "largest_contour",
    "segment_card",
    "color_masks",
]
