"""Tests de la vision del bloque de seguridad.

Se prueban sobre las 16 tarjetas reales de `assets/cards/`, que es material del
proyecto y no un sustituto sintetico: si alguien cambia los rangos HSV, el
umbral de circularidad o la morfologia y deja de reconocerse alguna tarjeta,
estos tests lo dicen.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.config import DebugConfig, SecurityConfig, get_config
from lanzarote_tracker.security.card_reader import StableCardReader, read_card
from lanzarote_tracker.security.segmentation import (
    build_color_mask,
    clean_mask,
    largest_contour,
    segment_card,
)
from lanzarote_tracker.security.shape_classifier import (
    approximate_polygon,
    circularity,
    classify_shape,
    is_square,
)
from lanzarote_tracker.security.types import Card, Color, Shape

SHAPES = ("circle", "triangle", "square", "pentagon")
COLORS = ("red", "green", "blue", "yellow")


@pytest.fixture()
def config() -> SecurityConfig:
    """Configuracion de seguridad por defecto."""
    return get_config().security


def card_image(shape: str, color: str) -> np.ndarray:
    """Carga la imagen de una tarjeta generada.

    Args:
        shape: Nombre de la forma.
        color: Nombre del color.

    Returns:
        La imagen BGR de la tarjeta.
    """
    path = Path("assets/cards") / f"{shape}_{color}.png"
    image = cv2.imread(str(path))
    assert image is not None, f"falta {path}; generala con scripts/generate_cards.py"
    return image


# --- Las 16 tarjetas --------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("color", COLORS)
def test_every_card_is_recognised(shape: str, color: str, config: SecurityConfig) -> None:
    """Las 16 combinaciones de forma y color se reconocen correctamente.

    Args:
        shape: Forma esperada.
        color: Color esperado.
        config: Configuracion de seguridad.
    """
    card = read_card(card_image(shape, color), config)
    assert card is not None, f"no se ha reconocido {shape}/{color}"
    assert card.shape is Shape(shape)
    assert card.color is Color(color)


def test_an_empty_frame_has_no_card(config: SecurityConfig) -> None:
    """Un fotograma en blanco no produce ninguna lectura."""
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert read_card(blank, config) is None


def test_a_dark_frame_has_no_card(config: SecurityConfig) -> None:
    """Con la camara tapada tampoco se inventa una tarjeta."""
    dark = np.zeros((480, 640, 3), dtype=np.uint8)
    assert read_card(dark, config) is None


# --- Segmentacion por color -------------------------------------------------
def test_red_mask_uses_both_ranges(config: SecurityConfig) -> None:
    """La mascara del rojo une sus dos tramos de tono.

    Es la comprobacion de que el rojo no se pierde por la costura del circulo:
    se construyen dos pixeles, uno en cada extremo, y los dos tienen que entrar.
    """
    hsv = np.zeros((2, 1, 3), dtype=np.uint8)
    low_range, high_range = config.hsv_ranges["red"]
    hsv[0, 0] = (low_range[0][0] + 1, 220, 220)
    hsv[1, 0] = (high_range[0][0] + 1, 220, 220)

    mask = build_color_mask(hsv, config.hsv_ranges["red"])
    assert mask[0, 0] == 255, "falta el tramo de tono bajo del rojo"
    assert mask[1, 0] == 255, "falta el tramo de tono alto del rojo"


def test_build_color_mask_needs_a_range(config: SecurityConfig) -> None:
    """Sin rangos no se puede construir nada."""
    hsv = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        build_color_mask(hsv, ())


def test_morphology_removes_salt_noise(config: SecurityConfig) -> None:
    """La apertura se lleva los pixeles sueltos del fondo."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (60, 60), (140, 140), 255, thickness=-1)
    noisy = mask.copy()
    noisy[10, 10] = 255
    noisy[20, 180] = 255

    cleaned = clean_mask(noisy, config)
    assert cleaned[10, 10] == 0
    assert cleaned[20, 180] == 0
    assert cleaned[100, 100] == 255, "la figura no deberia desaparecer"


def test_small_blobs_are_rejected(config: SecurityConfig) -> None:
    """Un contorno por debajo del area minima no cuenta como tarjeta."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (25, 25), 255, thickness=-1)
    assert largest_contour(mask, config.min_contour_area) is None


def test_the_biggest_contour_wins(config: SecurityConfig) -> None:
    """Entre dos manchas del mismo color gana la mayor."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (90, 90), 255, thickness=-1)
    cv2.rectangle(mask, (150, 150), (380, 380), 255, thickness=-1)
    contour = largest_contour(mask, config.min_contour_area)
    assert contour is not None
    assert cv2.contourArea(contour) > 40000


def test_segment_card_returns_the_right_colour(config: SecurityConfig) -> None:
    """De una tarjeta azul sale el color azul, no otro."""
    found = segment_card(card_image("square", "blue"), config)
    assert found is not None
    assert found[0] is Color.BLUE


# --- Clasificacion de forma -------------------------------------------------
@pytest.mark.parametrize(
    "shape,expected_vertices", [("triangle", 3), ("square", 4), ("pentagon", 5)]
)
def test_polygons_are_recognised_by_their_vertices(
    shape: str, expected_vertices: int, config: SecurityConfig
) -> None:
    """Los poligonos dan el numero de vertices que les corresponde.

    Args:
        shape: Forma a comprobar.
        expected_vertices: Vertices esperados.
        config: Configuracion de seguridad.
    """
    found = segment_card(card_image(shape, "green"), config)
    assert found is not None
    assert len(approximate_polygon(found[1], config)) == expected_vertices


def test_the_circle_has_more_vertices_than_a_pentagon(config: SecurityConfig) -> None:
    """approxPolyDP no sirve para el circulo: le asigna mas vertices.

    Por eso el circulo se decide por circularidad y no por conteo.
    """
    circle = segment_card(card_image("circle", "green"), config)
    pentagon = segment_card(card_image("pentagon", "green"), config)
    assert circle is not None and pentagon is not None
    assert len(approximate_polygon(circle[1], config)) > len(
        approximate_polygon(pentagon[1], config)
    )


def test_the_circle_is_the_roundest(config: SecurityConfig) -> None:
    """El circulo tiene la circularidad mas alta de las cuatro figuras."""
    values = {}
    for shape in SHAPES:
        found = segment_card(card_image(shape, "green"), config)
        assert found is not None
        values[shape] = circularity(found[1])
    assert values["circle"] == max(values.values())


def test_the_threshold_sits_between_square_and_circle(
    config: SecurityConfig,
) -> None:
    """El umbral separa el circulo del cuadrado, que es el caso mas apretado.

    Medido: cuadrado 0.791, circulo 0.897. Si alguien sube el umbral por encima
    del circulo o lo baja por debajo del cuadrado, el circulo deja de
    reconocerse o el cuadrado pasa por circulo.
    """
    square = segment_card(card_image("square", "green"), config)
    circle = segment_card(card_image("circle", "green"), config)
    assert square is not None and circle is not None
    assert circularity(square[1]) < config.circularity_threshold
    assert circularity(circle[1]) >= config.circularity_threshold


def test_a_rectangle_is_not_a_square(config: SecurityConfig) -> None:
    """Un rectangulo alargado con cuatro vertices se rechaza."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(mask, (20, 150), (380, 250), 255, thickness=-1)
    contour = largest_contour(mask, config.min_contour_area)
    assert contour is not None
    assert is_square(contour, config) is False
    assert classify_shape(contour, config) is None


def test_a_square_passes_the_aspect_check(config: SecurityConfig) -> None:
    """Un cuadrado de verdad pasa la comprobacion de relacion de aspecto."""
    found = segment_card(card_image("square", "red"), config)
    assert found is not None
    assert is_square(found[1], config) is True


# --- Validacion temporal ----------------------------------------------------
def feed(reader: StableCardReader, image: np.ndarray, times: int) -> List[Card]:
    """Entrega el mismo fotograma varias veces al lector.

    Args:
        reader: Lector de tarjetas.
        image: Fotograma a repetir.
        times: Numero de repeticiones.

    Returns:
        Las tarjetas confirmadas durante esas repeticiones.
    """
    confirmed = []
    for _ in range(times):
        card = reader.update(image)
        if card is not None:
            confirmed.append(card)
    return confirmed


def test_a_card_needs_n_consecutive_frames(config: SecurityConfig) -> None:
    """Antes de los N fotogramas no se confirma nada."""
    reader = StableCardReader(config)
    assert feed(reader, card_image("circle", "red"), config.consecutive_frames - 1) == []


def test_the_card_is_confirmed_exactly_at_n(config: SecurityConfig) -> None:
    """La confirmacion llega justo en el fotograma N."""
    reader = StableCardReader(config)
    confirmed = feed(reader, card_image("circle", "red"), config.consecutive_frames)
    assert len(confirmed) == 1
    assert confirmed[0].key == ("circle", "red")


def test_holding_the_card_does_not_confirm_it_again(config: SecurityConfig) -> None:
    """Una tarjeta quieta es UNA lectura, no muchas.

    Si cada fotograma extra contase, mantener una tarjeta delante de la camara
    completaria la secuencia entera con una sola.
    """
    reader = StableCardReader(config)
    confirmed = feed(reader, card_image("circle", "red"), config.consecutive_frames * 3)
    assert len(confirmed) == 1


def test_changing_the_card_restarts_the_count(config: SecurityConfig) -> None:
    """Cambiar de tarjeta a mitad reinicia el contador."""
    reader = StableCardReader(config)
    feed(reader, card_image("circle", "red"), config.consecutive_frames - 2)
    reader.update(card_image("square", "blue"))
    assert reader.stable_frames == 1


def test_an_empty_frame_breaks_the_streak(config: SecurityConfig) -> None:
    """Un fotograma sin tarjeta corta la racha y cuenta como hueco."""
    reader = StableCardReader(config)
    feed(reader, card_image("circle", "red"), config.consecutive_frames - 2)
    reader.update(np.full((480, 640, 3), 255, dtype=np.uint8))
    assert reader.stable_frames == 0
    assert reader.empty_frames == 1


def test_progress_reaches_one_when_confirmed(config: SecurityConfig) -> None:
    """La barra de progreso llega al final justo al confirmar."""
    reader = StableCardReader(config)
    feed(reader, card_image("circle", "red"), config.consecutive_frames)
    assert reader.progress == pytest.approx(1.0)


def test_reset_clears_the_reader(config: SecurityConfig) -> None:
    """reset deja el lector como recien creado."""
    reader = StableCardReader(config)
    feed(reader, card_image("circle", "red"), 3)
    reader.reset()
    assert reader.current_card is None
    assert reader.stable_frames == 0
    assert reader.empty_frames == 0


# --- Secuencia de transformacion para el informe ----------------------------
def test_the_debug_recorder_captures_the_pipeline(tmp_path: Path) -> None:
    """Con la depuracion activada se registran los pasos intermedios.

    Es la figura que exige el informe: original, HSV, mascaras y contorno.
    """
    config = get_config()
    debug = DebugConfig(enabled=True, output_root=tmp_path)
    recorder = DebugRecorder("security", debug)

    read_card(card_image("triangle", "yellow"), config.security, recorder=recorder)
    names = [name for name, _ in recorder.steps]

    assert "original" in names
    assert "hsv" in names
    assert any(name.startswith("mask_clean_") for name in names)
    assert any(name.startswith("contour_") for name in names)
    assert recorder.save_all(), "no se ha escrito ninguna imagen"
