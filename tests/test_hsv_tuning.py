"""Tests de la deduccion de rangos HSV a partir de una muestra.

El caso que importa es el rojo: su tono cruza la costura del circulo cromatico
y si no se detecta sale un rango que abarca todos los colores.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import pytest

from lanzarote_tracker.config import MAX_HUE, get_config
from lanzarote_tracker.security.hsv_tuning import (
    format_ranges,
    hue_bounds,
    sample_hsv_range,
)


def make_patch(hue: int, saturation: int = 230, value: int = 220, size: int = 15,
               noise: int = 4) -> np.ndarray:
    """Genera una muestra HSV alrededor de un tono, con algo de ruido.

    Args:
        hue: Tono central, de 0 a 179.
        saturation: Saturacion central.
        value: Valor central.
        size: Lado de la muestra.
        noise: Amplitud del ruido en cada canal.

    Returns:
        La muestra HSV de tres canales.
    """
    rng = np.random.default_rng(0)
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    hues = (hue + rng.integers(-noise, noise + 1, (size, size))) % (MAX_HUE + 1)
    patch[..., 0] = hues.astype(np.uint8)
    patch[..., 1] = np.clip(saturation + rng.integers(-noise, noise + 1, (size, size)), 0, 255)
    patch[..., 2] = np.clip(value + rng.integers(-noise, noise + 1, (size, size)), 0, 255)
    return patch


def in_range(hue: int, ranges: Tuple) -> bool:
    """Indica si un tono cae dentro de alguno de los rangos.

    Args:
        hue: Tono a comprobar.
        ranges: Rangos devueltos por sample_hsv_range.

    Returns:
        True si algun rango lo contiene.
    """
    return any(low[0] <= hue <= high[0] for low, high in ranges)


# --- Colores que no cruzan la costura ---------------------------------------
@pytest.mark.parametrize("hue", [60, 100, 25])
def test_a_plain_colour_gives_one_range(hue: int) -> None:
    """Verde, azul y amarillo se describen con un solo rango.

    Args:
        hue: Tono central de la muestra.
    """
    ranges = sample_hsv_range(make_patch(hue))
    assert len(ranges) == 1
    assert in_range(hue, ranges)


def test_the_range_contains_the_sampled_colour() -> None:
    """El tono muestreado cae dentro del rango que se devuelve."""
    ranges = sample_hsv_range(make_patch(60))
    (low, high) = ranges[0]
    assert low[0] <= 60 <= high[0]


def test_hue_bounds_stay_inside_opencv_limits() -> None:
    """Los limites nunca se salen de 0..179, que es el rango de OpenCV."""
    for hue in (0, 5, 90, 175, MAX_HUE):
        ranges = sample_hsv_range(make_patch(hue))
        for low, high in ranges:
            assert 0 <= low[0] <= MAX_HUE
            assert 0 <= high[0] <= MAX_HUE


def test_saturation_and_value_are_only_bounded_below() -> None:
    """Por arriba se deja 255: mas saturado o mas iluminado sigue siendo el color."""
    (low, high) = sample_hsv_range(make_patch(60))[0]
    assert high[1] == 255
    assert high[2] == 255
    assert low[1] > 0
    assert low[2] > 0


def test_a_dark_sample_lowers_the_value_floor() -> None:
    """Una muestra en penumbra admite valores mas bajos que una bien iluminada."""
    bright = sample_hsv_range(make_patch(60, value=240))[0][0]
    dark = sample_hsv_range(make_patch(60, value=90))[0][0]
    assert dark[2] < bright[2]


# --- El rojo, que si cruza la costura ---------------------------------------
def test_red_gives_two_ranges() -> None:
    """Una muestra con tonos a ambos lados de la costura devuelve dos rangos.

    Es el caso del rojo: pixeles cerca de 0 y pixeles cerca de 179.
    """
    patch = make_patch(0, noise=6)
    ranges = sample_hsv_range(patch)
    assert len(ranges) == 2


def test_red_ranges_cover_both_sides_of_the_seam() -> None:
    """Un tramo llega hasta 179 y el otro empieza en 0."""
    ranges = sample_hsv_range(make_patch(0, noise=6))
    highs = {high[0] for _, high in ranges}
    lows = {low[0] for low, _ in ranges}
    assert MAX_HUE in highs
    assert 0 in lows


def test_red_ranges_contain_the_red_hues() -> None:
    """Los tonos rojos de los dos lados caen dentro de los rangos."""
    ranges = sample_hsv_range(make_patch(0, noise=6))
    assert in_range(0, ranges)
    assert in_range(MAX_HUE, ranges)


def test_red_ranges_do_not_swallow_every_colour() -> None:
    """Los dos rangos juntos NO pueden abarcar el circulo entero.

    Es justo el fallo que se evita: tomar percentiles sin detectar la costura
    da un rango de 0 a 179, que deja pasar cualquier color.
    """
    ranges = sample_hsv_range(make_patch(0, noise=6))
    assert not in_range(60, ranges), "el verde no deberia entrar en el rojo"
    assert not in_range(100, ranges), "el azul no deberia entrar en el rojo"


def test_hue_bounds_reports_the_wrap() -> None:
    """hue_bounds avisa de que el rango cruza la costura."""
    hues = np.array([0, 1, 2, 177, 178, MAX_HUE] * 10)
    low, high, wrapped = hue_bounds(hues)
    assert wrapped is True
    assert low > high, "al envolver, el inferior queda por encima del superior"


def test_a_plain_colour_does_not_report_a_wrap() -> None:
    """Un color normal no activa la deteccion de costura."""
    _, _, wrapped = hue_bounds(np.full(100, 60))
    assert wrapped is False


# --- Entradas invalidas -----------------------------------------------------
def test_an_empty_sample_is_reported() -> None:
    """Un clic fuera de la imagen no puede medir nada."""
    with pytest.raises(ValueError, match="vacia"):
        sample_hsv_range(np.zeros((0, 0, 3), dtype=np.uint8))


def test_a_grayscale_sample_is_rejected() -> None:
    """La muestra tiene que venir ya convertida a HSV."""
    with pytest.raises(ValueError, match="tres canales"):
        sample_hsv_range(np.zeros((5, 5), dtype=np.uint8))


# --- Formato de salida ------------------------------------------------------
def test_format_matches_the_configuration_syntax() -> None:
    """La linea que se imprime se puede pegar tal cual en config.py."""
    ranges = (((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255)))
    line = format_ranges("red", ranges)
    assert line.strip().startswith('"red":')
    assert "((0, 120, 70), (10, 255, 255))" in line
    assert "((170, 120, 70), (179, 255, 255))" in line


def test_a_single_range_keeps_the_trailing_comma() -> None:
    """Un solo rango necesita la coma final para seguir siendo una tupla."""
    line = format_ranges("green", (((36, 80, 70), (89, 255, 255)),))
    assert line.rstrip().endswith("),),")


# --- Coherencia con lo que el proyecto ya tiene configurado -----------------
def test_the_generated_cards_are_measured_correctly() -> None:
    """Midiendo una tarjeta generada se recupera un rango que la contiene.

    Cierra el circulo: `generate_cards.py` dibuja con el color central del
    rango configurado y aqui se mide ese mismo color.
    """
    config = get_config()
    for color in ("green", "blue", "yellow"):
        (low, high) = config.security.hsv_ranges[color][0]
        hue = (low[0] + high[0]) // 2
        bgr = cv2.cvtColor(np.uint8([[[hue, 235, 235]]]), cv2.COLOR_HSV2BGR)
        patch = cv2.cvtColor(np.tile(bgr, (15, 15, 1)), cv2.COLOR_BGR2HSV)
        ranges = sample_hsv_range(patch)
        assert in_range(hue, ranges), f"el rango medido no contiene el tono de {color}"


def test_a_red_near_the_seam_wraps_even_if_the_sample_does_not() -> None:
    """Un rojo en tono 5 envuelve, porque el margen cruza la costura.

    Es el caso de las tarjetas generadas: todos sus pixeles rojos estan en
    tono 5, asi que la muestra no cruza nada. Pero el rango util SI tiene que
    cruzar: un rojo impreso o en pantalla tiene pixeles en 177, 178 y 179, y
    recortar el margen en 0 los dejaria fuera en silencio.
    """
    ranges = sample_hsv_range(make_patch(5, noise=0))
    assert len(ranges) == 2
    assert in_range(MAX_HUE, ranges), "faltan los tonos del otro lado de la costura"
    assert in_range(0, ranges)
    assert not in_range(60, ranges)


def test_the_generated_red_card_is_measured_as_two_ranges() -> None:
    """Midiendo la tarjeta roja real salen los dos rangos que espera config."""
    img = cv2.imread("assets/cards/circle_red.png")
    assert img is not None, "falta assets/cards/circle_red.png"
    height, width = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    patch = hsv[height // 2 - 7 : height // 2 + 8, width // 2 - 7 : width // 2 + 8]
    ranges = sample_hsv_range(patch)
    assert len(ranges) == 2
    assert in_range(0, ranges)
    assert in_range(MAX_HUE, ranges)


def test_a_huge_margin_gives_up_instead_of_wrapping_nonsense() -> None:
    """Si el margen abarca el circulo entero, se devuelve completo y sin envolver.

    Mejor un rango inutil pero coherente que dos tramos que se solapan.
    """
    ranges = sample_hsv_range(make_patch(60), margin=(120, 60, 60))
    assert len(ranges) == 1
    assert ranges[0][0][0] == 0
    assert ranges[0][1][0] == MAX_HUE
