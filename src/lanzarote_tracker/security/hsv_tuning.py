"""Deduccion de un rango HSV a partir de una muestra de pixeles.

Ajustar los rangos moviendo seis deslizadores a ojo es lento y poco
reproducible: cada uno para donde le parece y el resultado no se puede
justificar en el informe. Aqui se hace al reves: se toma un trozo de la
tarjeta y se calcula el rango que contiene a la mayoria de sus pixeles.

EL PROBLEMA DEL ROJO. En OpenCV el tono H va de 0 a 179 y es CIRCULAR: el rojo
esta justo en la costura, con pixeles cerca de 0 y cerca de 179. Si se toman
percentiles sin mas sobre esos valores sale un rango de 0 a 179, es decir,
todos los colores. Por eso este modulo detecta la envoltura y devuelve DOS
rangos, que es exactamente lo que `SecurityConfig.hsv_ranges` espera para el
rojo.

Practica 1: es el mismo criterio de umbralizacion por rango de tono que el
`get_hsv_color_ranges` de la practica, pero midiendo la muestra en vez de
ajustando a mano.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from lanzarote_tracker.config import MAX_HUE, HsvRange

#: Margen por defecto que se anade al rango medido, en (H, S, V). Generoso en
#: saturacion y valor porque son los que mas cambian con la luz; estrecho en
#: tono porque es lo que distingue un color de otro.
DEFAULT_MARGIN: Tuple[int, int, int] = (8, 60, 60)

#: Si el tono medido abarca mas de esto, se asume que cruza la costura del
#: circulo cromatico. Ningun color de tarjeta ocupa medio circulo de verdad.
WRAP_THRESHOLD: int = 90


def _percentiles(values: np.ndarray, percentile: float) -> Tuple[int, int]:
    """Devuelve los percentiles inferior y superior de un canal.

    Se usan percentiles y no el minimo y el maximo para que un par de pixeles
    de borde, donde la tarjeta se mezcla con el fondo, no estiren el rango.

    Args:
        values: Valores del canal, aplanados.
        percentile: Percentil inferior, en tanto por ciento.

    Returns:
        El par (inferior, superior), redondeados a entero.
    """
    low = int(round(float(np.percentile(values, percentile))))
    high = int(round(float(np.percentile(values, 100.0 - percentile))))
    return low, high


def hue_bounds(
    hues: np.ndarray,
    percentile: float = 5.0,
    margin: int = DEFAULT_MARGIN[0],
) -> Tuple[int, int, bool]:
    """Calcula los limites de tono de una muestra, teniendo en cuenta la costura.

    Args:
        hues: Valores del canal H, de 0 a 179.
        percentile: Percentil inferior que se descarta por cada lado.
        margin: Margen que se anade a cada lado del rango medido.

    Returns:
        Una tupla (inferior, superior, envuelve). Si `envuelve` es True, el
        rango va del inferior hasta 179 y sigue de 0 hasta el superior, de modo
        que el inferior es MAYOR que el superior.
    """
    values = np.asarray(hues).astype(np.int32).ravel()
    span = MAX_HUE + 1
    low, high = _percentiles(values, percentile)

    if high - low > WRAP_THRESHOLD:
        # La muestra YA cruza la costura: se gira medio circulo, donde no la
        # cruza, se miden los limites y se deshace el giro. El inferior puede
        # quedar negativo, que es justo lo que indica que envuelve.
        shifted = (values + span // 2) % span
        shifted_low, shifted_high = _percentiles(shifted, percentile)
        low = shifted_low - span // 2
        high = shifted_high - span // 2

    low -= margin
    high += margin

    if high - low >= MAX_HUE:
        # El margen se ha comido el circulo entero: el rango deja de distinguir
        # nada, asi que se devuelve completo y sin envolver.
        return 0, MAX_HUE, False

    # Ojo: aqui NO se puede recortar a 0..179. Un rojo medido en tono 5 con
    # margen 8 va de -3 a 13, y -3 no es 0: es 177. Recortarlo perderia justo
    # los tonos del otro lado de la costura que todo rojo real tiene.
    wrapped = low < 0 or high > MAX_HUE
    return int(low % span), int(high % span), wrapped


def sample_hsv_range(
    patch: np.ndarray,
    percentile: float = 5.0,
    margin: Tuple[int, int, int] = DEFAULT_MARGIN,
) -> Tuple[HsvRange, ...]:
    """Deduce el rango HSV que describe un trozo de tarjeta.

    La saturacion y el valor solo se acotan por abajo: por arriba se deja 255.
    Un color mas saturado o mas iluminado de lo medido sigue siendo el mismo
    color, mientras que por debajo estan los grises y las sombras, que son
    justo lo que hay que dejar fuera.

    Args:
        patch: Trozo de imagen YA CONVERTIDO A HSV, de cualquier tamano.
        percentile: Percentil inferior que se descarta por cada lado, para que
            los pixeles del borde de la figura no estiren el rango.
        margin: Margen en (H, S, V) que se anade al rango medido.

    Returns:
        Uno o dos rangos, listos para pegar en `SecurityConfig.hsv_ranges`. Son
        DOS cuando el tono cruza la costura del circulo cromatico, que es el
        caso del rojo.

    Raises:
        ValueError: Si la muestra esta vacia o no tiene tres canales.
    """
    array = np.asarray(patch)
    if array.size == 0:
        raise ValueError("La muestra esta vacia: haz clic sobre la figura.")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("La muestra debe ser una imagen HSV de tres canales.")

    low_hue, high_hue, wrapped = hue_bounds(array[..., 0], percentile, margin[0])

    sat_low, _ = _percentiles(array[..., 1].ravel(), percentile)
    val_low, _ = _percentiles(array[..., 2].ravel(), percentile)
    sat_low = max(0, sat_low - margin[1])
    val_low = max(0, val_low - margin[2])

    if not wrapped:
        return (((low_hue, sat_low, val_low), (high_hue, 255, 255)),)

    # Dos tramos, en orden ascendente de tono para que coincidan con como se
    # escribe el rojo en `DEFAULT_HSV_RANGES`: primero el que arranca en 0 y
    # despues el que termina en 179. El orden no afecta a la mascara, que los
    # une con un OR, pero asi la linea que se imprime se pega tal cual.
    return (
        ((0, sat_low, val_low), (high_hue, 255, 255)),
        ((low_hue, sat_low, val_low), (MAX_HUE, 255, 255)),
    )


def format_ranges(color: str, ranges: Tuple[HsvRange, ...]) -> str:
    """Formatea unos rangos como la linea que hay que pegar en `config.py`.

    Args:
        color: Nombre del color, p. ej. "red".
        ranges: Rangos devueltos por :func:`sample_hsv_range`.

    Returns:
        Una linea lista para copiar dentro de `DEFAULT_HSV_RANGES`.
    """
    parts = ", ".join(
        f"(({low[0]}, {low[1]}, {low[2]}), ({high[0]}, {high[1]}, {high[2]}))"
        for low, high in ranges
    )
    suffix = "," if len(ranges) == 1 else ""
    return f'    "{color}": ({parts}{suffix}),'


__all__ = [
    "DEFAULT_MARGIN",
    "WRAP_THRESHOLD",
    "hue_bounds",
    "sample_hsv_range",
    "format_ranges",
]
