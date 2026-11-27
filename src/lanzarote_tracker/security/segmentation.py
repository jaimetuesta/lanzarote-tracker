"""Segmentacion de las tarjetas por color.

Pipeline por fotograma, en este orden:

1. Correccion de distorsion con la calibracion de la webcam (Practica 2), que
   aplica quien llama antes de pasar el fotograma.
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

from typing import Dict, Optional, Tuple

import cv2
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

    Raises:
        ValueError: Si no se pasa ningun rango.
    """
    if not ranges:
        raise ValueError("Hace falta al menos un rango HSV para construir la mascara.")

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        piece = cv2.inRange(
            hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)
        )
        mask = cv2.bitwise_or(mask, piece)
    return mask


def clean_mask(mask: Mask, config: SecurityConfig) -> Mask:
    """Limpia una mascara con apertura y cierre morfologicos.

    Practica 3: cv2.morphologyEx con MORPH_OPEN (quita el ruido de sal) y
    despues MORPH_CLOSE (cierra los huecos dentro de la tarjeta), con el
    elemento estructurante y el numero de iteraciones de la configuracion.

    OJO con pasarse de kernel o de iteraciones: la morfologia REDONDEA LAS
    ESQUINAS, y de las esquinas depende el clasificador de forma. Un cierre
    agresivo convierte un pentagono en algo que mide como un circulo. Los
    valores de la configuracion estan elegidos para limpiar sin deformar.

    Args:
        mask: Mascara binaria de entrada, que no se modifica.
        config: Configuracion de seguridad, de la que se toman el tamano del
            kernel y las iteraciones.

    Returns:
        Una mascara nueva, limpia.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.morph_kernel_size)
    opened = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iterations
    )
    return cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iterations
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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area:
        return None
    return biggest


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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if recorder is not None:
        recorder.record("original", frame)
        recorder.record("hsv", hsv)

    masks: Dict[Color, Mask] = {}
    for name, ranges in config.hsv_ranges.items():
        raw = build_color_mask(hsv, ranges)
        cleaned = clean_mask(raw, config)
        masks[Color(name)] = cleaned
        if recorder is not None:
            recorder.record(f"mask_raw_{name}", raw)
            recorder.record(f"mask_clean_{name}", cleaned)
    return masks


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
            y el contorno ganador dibujado.

    Returns:
        El par (color, contorno) de la tarjeta, o None si no hay ninguna tarjeta
        valida en el fotograma.
    """
    masks = color_masks(frame, config, recorder=recorder)

    best_area = 0.0
    best: Optional[Tuple[Color, Contour]] = None
    for color, mask in masks.items():
        contour = largest_contour(mask, config.min_contour_area)
        if contour is None:
            continue
        area = float(cv2.contourArea(contour))
        if area > best_area:
            best_area, best = area, (color, contour)

    if best is not None and recorder is not None:
        drawn = frame.copy()
        cv2.drawContours(drawn, [best[1]], -1, (0, 255, 0), 3)
        recorder.record(f"contour_{best[0].value}", drawn)
    return best


__all__ = [
    "Mask",
    "Contour",
    "build_color_mask",
    "clean_mask",
    "largest_contour",
    "segment_card",
    "color_masks",
]
