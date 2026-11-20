"""Tests del modulo de calibracion.

No hacen falta ni camara ni fotos reales: el tablero se genera sinteticamente
con NumPy, asi que la deteccion de esquinas se prueba de verdad y no solo la
aritmetica de los puntos 3D.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

from lanzarote_tracker.calibration.calibrate import (
    CalibrationResult,
    aggregate_rms,
    calibrate_camera,
    reprojection_errors,
    run_calibration,
)
from lanzarote_tracker.calibration.chessboard import (
    build_object_points,
    collect_corners,
    detect_pattern_size,
    find_corners,
    parse_pattern_size,
    to_gray,
)
from lanzarote_tracker.calibration.store import (
    load_calibration,
    save_calibration,
    try_load_calibration,
)
from lanzarote_tracker.calibration.undistort import (
    Undistorter,
    side_by_side,
    undistort_image,
)
from lanzarote_tracker.config import CalibrationConfig, PathsConfig, get_config

#: Tablero sintetico: 9x6 esquinas internas son 10x7 cuadrados.
PATTERN: Tuple[int, int] = (9, 6)
SQUARE_PX: int = 40
#: Margen blanco alrededor del tablero. cv2.findChessboardCorners lo necesita:
#: sin borde no distingue los cuadrados del extremo del fondo.
MARGIN_PX: int = 60


def make_chessboard(
    pattern: Tuple[int, int] = PATTERN,
    square: int = SQUARE_PX,
    margin: int = MARGIN_PX,
) -> np.ndarray:
    """Genera una imagen sintetica de un tablero de ajedrez.

    Args:
        pattern: Esquinas internas (columnas, filas).
        square: Lado de cada cuadrado, en pixeles.
        margin: Margen blanco alrededor del tablero, en pixeles.

    Returns:
        La imagen BGR del tablero, sobre fondo blanco.
    """
    columns, rows = pattern
    board_columns, board_rows = columns + 1, rows + 1
    board = np.zeros((board_rows * square, board_columns * square), dtype=np.uint8)
    for row in range(board_rows):
        for column in range(board_columns):
            if (row + column) % 2 == 0:
                board[
                    row * square : (row + 1) * square,
                    column * square : (column + 1) * square,
                ] = 255
    canvas = np.full(
        (board.shape[0] + 2 * margin, board.shape[1] + 2 * margin), 255, dtype=np.uint8
    )
    canvas[margin : margin + board.shape[0], margin : margin + board.shape[1]] = board
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


@pytest.fixture()
def config() -> CalibrationConfig:
    """Configuracion de calibracion ajustada al tablero sintetico."""
    return replace(get_config().calibration, pattern_size=PATTERN, square_size_mm=25.0)


@pytest.fixture()
def board() -> np.ndarray:
    """Imagen sintetica del tablero."""
    return make_chessboard()


# --- parse_pattern_size -----------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [("9x6", (9, 6)), ("9X6", (9, 6)), ("9,6", (9, 6)), (" 7 5 ", (7, 5))],
)
def test_parse_pattern_size_accepts_several_formats(
    text: str, expected: Tuple[int, int]
) -> None:
    """Se admite el tamano escrito con x, con coma o con espacio.

    Args:
        text: Texto de entrada.
        expected: Tamano esperado.
    """
    assert parse_pattern_size(text) == expected


@pytest.mark.parametrize("text", ["9", "9x", "axb", "9x6x3", "1x6", "9x1", ""])
def test_parse_pattern_size_rejects_invalid_text(text: str) -> None:
    """Un tamano mal escrito o degenerado se rechaza con ValueError.

    Args:
        text: Texto invalido.
    """
    with pytest.raises(ValueError):
        parse_pattern_size(text)


# --- build_object_points ----------------------------------------------------
def test_object_points_have_one_row_per_corner(config: CalibrationConfig) -> None:
    """Hay un punto 3D por esquina interna del tablero."""
    points = build_object_points(config)
    columns, rows = config.pattern_size
    assert points.shape == (columns * rows, 3)
    assert points.dtype == np.float32


def test_object_points_are_planar(config: CalibrationConfig) -> None:
    """El tablero se modela plano: todos los puntos tienen Z = 0."""
    assert np.all(build_object_points(config)[:, 2] == 0.0)


def test_object_points_are_spaced_by_the_square_size(
    config: CalibrationConfig,
) -> None:
    """Las esquinas contiguas estan separadas por el lado del cuadrado."""
    config = replace(config, square_size_mm=25.0)
    points = build_object_points(config)
    assert points[1, 0] - points[0, 0] == pytest.approx(25.0)


def test_square_size_only_scales_the_points(config: CalibrationConfig) -> None:
    """Doblar el lado del cuadrado dobla las coordenadas: es solo una escala.

    Por eso el lado del cuadrado no afecta a K ni a los coeficientes de
    distorsion, que es el unico uso que da este proyecto a la calibracion.
    """
    single = build_object_points(replace(config, square_size_mm=25.0))
    double = build_object_points(replace(config, square_size_mm=50.0))
    assert np.allclose(double, single * 2.0)


# --- find_corners -----------------------------------------------------------
def test_find_corners_on_synthetic_board(
    board: np.ndarray, config: CalibrationConfig
) -> None:
    """Se encuentran todas las esquinas internas del tablero sintetico."""
    corners = find_corners(board, config)
    assert corners is not None
    columns, rows = config.pattern_size
    assert corners.shape == (columns * rows, 1, 2)


def test_find_corners_returns_none_without_a_board(
    config: CalibrationConfig,
) -> None:
    """Sin tablero a la vista se devuelve None, no una excepcion."""
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert find_corners(blank, config) is None


def test_find_corners_returns_none_with_the_wrong_pattern(
    board: np.ndarray, config: CalibrationConfig
) -> None:
    """Un `pattern_size` equivocado no encuentra nada: es un fallo de todo o nada."""
    wrong = replace(config, pattern_size=(5, 4))
    assert find_corners(board, wrong) is None


def test_to_gray_is_idempotent(board: np.ndarray) -> None:
    """to_gray acepta imagenes ya en escala de grises sin volver a convertirlas."""
    gray = to_gray(board)
    assert gray.ndim == 2
    assert to_gray(gray) is gray


# --- detect_pattern_size ----------------------------------------------------
def test_detect_pattern_size_finds_the_real_board(board: np.ndarray) -> None:
    """El mayor tamano que encaja es el del tablero de verdad.

    Un tablero siempre encaja tambien con sus subrejillas, por eso se toma el
    mayor.
    """
    matches = detect_pattern_size(board, max_size=10)
    assert matches
    assert matches[0] == PATTERN


# --- collect_corners --------------------------------------------------------
def test_collect_corners_discards_images_without_board(
    board: np.ndarray, config: CalibrationConfig
) -> None:
    """Las fotos sin tablero se descartan en vez de romper la calibracion."""
    blank = np.full(board.shape, 255, dtype=np.uint8)
    config = replace(config, min_images=2)
    object_points, image_points, size = collect_corners([board, blank, board], config)
    assert len(object_points) == len(image_points) == 2
    assert size == (board.shape[1], board.shape[0])


def test_collect_corners_requires_the_minimum(
    board: np.ndarray, config: CalibrationConfig
) -> None:
    """Por debajo de `min_images` se avisa con un ValueError explicativo."""
    config = replace(config, min_images=5)
    with pytest.raises(ValueError, match="ESQUINAS INTERNAS"):
        collect_corners([board], config)


def test_collect_corners_rejects_mixed_sizes(
    board: np.ndarray, config: CalibrationConfig
) -> None:
    """Mezclar resoluciones se rechaza: la calibracion vale para un solo tamano."""
    smaller = cv2.resize(board, (board.shape[1] // 2, board.shape[0] // 2))
    config = replace(config, min_images=1)
    with pytest.raises(ValueError, match="mismo tamano"):
        collect_corners([board, smaller], config)


def test_collect_corners_rejects_an_empty_list(config: CalibrationConfig) -> None:
    """Sin imagenes no hay nada que recoger."""
    with pytest.raises(ValueError):
        collect_corners([], config)


# --- calibrate_camera -------------------------------------------------------
def test_calibrate_camera_needs_enough_images(
    board: np.ndarray, config: CalibrationConfig
) -> None:
    """La calibracion exige el minimo de fotos validas configurado."""
    config = replace(config, min_images=10)
    with pytest.raises(ValueError):
        calibrate_camera([board, board], config)


def test_run_calibration_reports_an_empty_directory(
    tmp_path: Path, config: CalibrationConfig
) -> None:
    """Un directorio sin fotos se avisa con FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_calibration(tmp_path, config)


# --- CalibrationResult ------------------------------------------------------
def make_result(distortion: float = 0.0) -> CalibrationResult:
    """Construye un resultado de calibracion sintetico.

    Args:
        distortion: Valor de k1, para poder probar con y sin distorsion.

    Returns:
        Un :class:`CalibrationResult` con una K plausible de 640x480.
    """
    camera_matrix = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    coefficients = np.array([[distortion, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    return CalibrationResult(
        camera_matrix=camera_matrix,
        distortion_coefficients=coefficients,
        rms=0.25,
        image_size=(640, 480),
        num_images=15,
    )


def test_result_exposes_focal_length_and_principal_point() -> None:
    """fx, fy, cx y cy se leen de la matriz K."""
    result = make_result()
    assert result.focal_length == (600.0, 600.0)
    assert result.principal_point == (320.0, 240.0)


def test_result_summary_mentions_the_rms() -> None:
    """El resumen incluye el RMS, que es lo primero que se mira."""
    assert "RMS" in make_result().summary()


def test_reprojection_errors_need_the_pose_vectors() -> None:
    """Sin rvec ni tvec no se puede calcular el error por imagen."""
    with pytest.raises(ValueError):
        reprojection_errors(make_result(), [], [])


# --- Normalizacion del error de reproyeccion --------------------------------
def make_posed_result(
    config: CalibrationConfig, views: int
) -> Tuple[CalibrationResult, list, list]:
    """Construye una calibracion sintetica con reproyeccion exacta.

    Proyecta los puntos del patron con una pose conocida y devuelve esas mismas
    proyecciones como esquinas detectadas, de modo que el error es nulo por
    construccion y cualquier desviacion que se anada despues es exactamente la
    que se quiera medir.

    Args:
        config: Configuracion de calibracion.
        views: Numero de vistas a generar.

    Returns:
        Una tupla (resultado, puntos 3D, puntos 2D).
    """
    result = make_result()
    points_3d = build_object_points(config)
    rotation = np.zeros((3, 1), dtype=np.float64)
    translation = np.array([[-100.0], [-80.0], [600.0]], dtype=np.float64)

    projected, _ = cv2.projectPoints(
        points_3d,
        rotation,
        translation,
        result.camera_matrix,
        result.distortion_coefficients,
    )
    posed = replace(
        result,
        rotation_vectors=tuple(rotation.copy() for _ in range(views)),
        translation_vectors=tuple(translation.copy() for _ in range(views)),
    )
    return (
        posed,
        [points_3d.copy() for _ in range(views)],
        [projected.copy() for _ in range(views)],
    )


def test_reprojection_error_is_zero_for_perfect_points(
    config: CalibrationConfig,
) -> None:
    """Si las esquinas coinciden con la proyeccion, el error es nulo."""
    result, object_points, image_points = make_posed_result(config, views=2)
    errors = reprojection_errors(result, object_points, image_points)
    assert errors == pytest.approx([0.0, 0.0], abs=1e-6)


def test_reprojection_error_equals_the_pixel_offset(
    config: CalibrationConfig,
) -> None:
    """Un desplazamiento conocido de 5 px da un RMS de exactamente 5 px.

    Fija la normalizacion: se divide por la raiz del numero de esquinas, no por
    el numero de esquinas. Con la division por N este valor saldria sqrt(54)
    veces menor y no seria comparable con el RMS de cv2.calibrateCamera.
    """
    result, object_points, image_points = make_posed_result(config, views=1)
    # Desplazamiento de (3, 4) px en cada esquina: distancia euclidea 5 px.
    image_points[0][:, 0, 0] += 3.0
    image_points[0][:, 0, 1] += 4.0
    errors = reprojection_errors(result, object_points, image_points)
    assert errors[0] == pytest.approx(5.0, abs=1e-6)


def test_aggregate_rms_is_the_quadratic_mean(config: CalibrationConfig) -> None:
    """El RMS global es la media cuadratica de los RMS por imagen.

    Con una imagen perfecta y otra con 5 px de error, el global es 5/sqrt(2).
    """
    result, object_points, image_points = make_posed_result(config, views=2)
    image_points[1][:, 0, 0] += 3.0
    image_points[1][:, 0, 1] += 4.0
    errors = reprojection_errors(result, object_points, image_points)
    assert aggregate_rms(errors) == pytest.approx(5.0 / np.sqrt(2.0), abs=1e-6)


def test_aggregate_rms_of_no_images_is_zero() -> None:
    """Sin imagenes el RMS global es 0.0 y no una division por cero."""
    assert aggregate_rms([]) == 0.0


# --- undistort --------------------------------------------------------------
def test_undistorter_without_distortion_keeps_the_image() -> None:
    """Con coeficientes nulos la correccion devuelve practicamente la misma imagen."""
    result = make_result(distortion=0.0)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    undistorted = Undistorter(result, alpha=1.0)(image, crop=False)
    assert undistorted.shape == image.shape
    assert np.abs(undistorted.astype(int) - image.astype(int)).mean() < 1.0


def test_undistorter_rejects_another_resolution() -> None:
    """Aplicar la calibracion a otra resolucion se rechaza con un aviso claro."""
    undistorter = Undistorter(make_result())
    other = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="misma resolucion"):
        undistorter(other)


def test_undistort_image_changes_a_distorted_camera() -> None:
    """Con distorsion no nula, la imagen corregida difiere de la original."""
    image = make_chessboard(square=20, margin=20)
    image = cv2.resize(image, (640, 480))
    corrected = undistort_image(image, make_result(distortion=-0.3), crop=False)
    assert corrected.shape == image.shape
    assert np.abs(corrected.astype(int) - image.astype(int)).mean() > 1.0


def test_side_by_side_puts_both_views_together() -> None:
    """La figura del informe junta original y corregida a la misma altura."""
    original = np.zeros((480, 640, 3), dtype=np.uint8)
    corrected = np.zeros((400, 500, 3), dtype=np.uint8)
    figure = side_by_side(original, corrected)
    assert figure.shape[0] == original.shape[0]
    assert figure.shape[1] > original.shape[1]


# --- store ------------------------------------------------------------------
def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """Lo que se guarda se recupera igual."""
    config = get_config().calibration
    paths = replace(PathsConfig(), calibration_results=tmp_path)
    result = make_result(distortion=-0.12)

    npz_path, json_path = save_calibration(result, paths, config)
    assert npz_path.is_file()
    assert json_path.is_file()

    loaded = load_calibration(paths, config)
    assert np.allclose(loaded.camera_matrix, result.camera_matrix)
    assert np.allclose(loaded.distortion_coefficients, result.distortion_coefficients)
    assert loaded.rms == pytest.approx(result.rms)
    assert loaded.image_size == result.image_size
    assert loaded.num_images == result.num_images


def test_saved_json_is_readable_for_the_report(tmp_path: Path) -> None:
    """El .json trae los datos que hay que pegar en el informe."""
    import json

    config = get_config().calibration
    paths = replace(PathsConfig(), calibration_results=tmp_path)
    _, json_path = save_calibration(make_result(), paths, config)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for key in (
        "camera_matrix",
        "distortion_coefficients",
        "rms",
        "image_size",
        "num_images",
        "pattern_size",
        "square_size_mm",
    ):
        assert key in payload


def test_load_calibration_reports_a_missing_file(tmp_path: Path) -> None:
    """Si no se ha calibrado todavia, se dice que hay que hacerlo."""
    paths = replace(PathsConfig(), calibration_results=tmp_path)
    with pytest.raises(FileNotFoundError, match="calibrate_camera"):
        load_calibration(paths, get_config().calibration)


def test_try_load_calibration_returns_none_when_missing(tmp_path: Path) -> None:
    """La variante tolerante devuelve None en vez de romper el pipeline."""
    paths = replace(PathsConfig(), calibration_results=tmp_path)
    assert try_load_calibration(paths, get_config().calibration) is None
