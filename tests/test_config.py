"""Tests de la configuracion centralizada.

Son tests reales: la configuracion ya esta implementada y no depende de vision,
asi que se puede comprobar entera. Protegen sobre todo los invariantes que es
facil romper al afinar parametros, como el tope del tono H en OpenCV o los dos
rangos del rojo.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from lanzarote_tracker.config import (
    MAX_HUE,
    AppConfig,
    DebugConfig,
    get_config,
    with_debug,
)
from lanzarote_tracker.security.types import Color, Shape


def test_get_config_returns_app_config() -> None:
    """get_config devuelve una AppConfig con todos los bloques."""
    config = get_config()
    assert isinstance(config, AppConfig)
    for block in (
        "paths",
        "camera",
        "calibration",
        "security",
        "capture",
        "detection",
        "tracking",
        "classification",
        "identification",
        "visualization",
        "debug",
    ):
        assert hasattr(config, block), f"Falta el bloque de configuracion '{block}'"


def test_config_is_frozen() -> None:
    """La configuracion es inmutable: un modulo no puede pisar la de otro."""
    config = get_config()
    with pytest.raises(FrozenInstanceError):
        config.security.consecutive_frames = 1  # type: ignore[misc]


def test_get_config_returns_independent_instances() -> None:
    """Dos llamadas devuelven objetos distintos, no un singleton compartido."""
    first = get_config()
    second = get_config()
    assert first is not second
    assert first.security.hsv_ranges is not second.security.hsv_ranges


# --- Seguridad --------------------------------------------------------------
def test_hue_values_within_opencv_range() -> None:
    """Todos los tonos H estan entre 0 y 179, que es el rango de OpenCV."""
    config = get_config()
    for color, ranges in config.security.hsv_ranges.items():
        for low, high in ranges:
            assert 0 <= low[0] <= MAX_HUE, f"H minimo fuera de rango en '{color}'"
            assert 0 <= high[0] <= MAX_HUE, f"H maximo fuera de rango en '{color}'"
            assert low[0] <= high[0], f"H minimo mayor que el maximo en '{color}'"


def test_saturation_and_value_within_range() -> None:
    """Saturacion y valor estan entre 0 y 255 y el minimo no supera al maximo."""
    config = get_config()
    for color, ranges in config.security.hsv_ranges.items():
        for low, high in ranges:
            for index in (1, 2):
                assert 0 <= low[index] <= 255, f"Canal {index} fuera de rango en '{color}'"
                assert 0 <= high[index] <= 255, f"Canal {index} fuera de rango en '{color}'"
                assert low[index] <= high[index], f"Minimo > maximo en '{color}'"


def test_red_needs_two_hue_ranges() -> None:
    """El rojo tiene DOS rangos: su tono esta partido por el extremo del circulo."""
    config = get_config()
    red_ranges = config.security.hsv_ranges[Color.RED.value]
    assert len(red_ranges) == 2, "El rojo necesita dos rangos HSV unidos"
    low_range, high_range = red_ranges
    assert low_range[0][0] < 20, "El primer rango del rojo debe tener H cerca de 0"
    assert high_range[1][0] >= MAX_HUE - 20, "El segundo debe llegar cerca de 179"


def test_non_red_colors_have_a_single_range() -> None:
    """El resto de colores se capturan con un solo rango de tono."""
    config = get_config()
    for color in (Color.GREEN, Color.BLUE, Color.YELLOW):
        assert len(config.security.hsv_ranges[color.value]) == 1


def test_every_card_color_has_hsv_ranges() -> None:
    """Hay rangos HSV configurados para los cuatro colores de tarjeta."""
    config = get_config()
    assert set(config.security.hsv_ranges) == {color.value for color in Color}


def test_expected_sequence_matches_sequence_length() -> None:
    """La secuencia esperada tiene exactamente `sequence_length` tarjetas."""
    security = get_config().security
    assert len(security.expected_sequence) == security.sequence_length


def test_expected_sequence_uses_known_shapes_and_colors() -> None:
    """Las tarjetas de la secuencia usan formas y colores reconocidos."""
    shapes = {shape.value for shape in Shape}
    colors = {color.value for color in Color}
    for shape, color in get_config().security.expected_sequence:
        assert shape in shapes, f"Forma desconocida en la secuencia: {shape}"
        assert color in colors, f"Color desconocido en la secuencia: {color}"


def test_security_timings_are_positive() -> None:
    """N, los huecos y los tiempos de espera y bloqueo son positivos."""
    security = get_config().security
    assert security.consecutive_frames > 0
    assert security.gap_frames > 0
    assert security.max_seconds_without_card > 0
    assert security.max_failed_attempts > 0
    assert security.lockout_seconds > 0


def test_vertices_by_shape_excludes_circle() -> None:
    """El circulo no se reconoce por vertices, sino por circularidad."""
    security = get_config().security
    assert Shape.CIRCLE.value not in security.vertices_by_shape
    assert security.vertices_by_shape[Shape.TRIANGLE.value] == 3
    assert security.vertices_by_shape[Shape.SQUARE.value] == 4
    assert security.vertices_by_shape[Shape.PENTAGON.value] == 5


def test_circularity_threshold_is_a_fraction() -> None:
    """El umbral de circularidad esta entre 0 y 1, como C = 4*pi*A/P^2."""
    threshold = get_config().security.circularity_threshold
    assert 0.0 < threshold <= 1.0


# --- Backends ---------------------------------------------------------------
def test_default_backends_are_the_classic_ones() -> None:
    """Los backends por defecto son los que NO dependen de deep learning."""
    config = get_config()
    assert config.detection.backend == "classic"
    assert config.classification.backend == "bow"


def test_coco_class_id_matches_its_convention() -> None:
    """El indice de `airplane` concuerda con la convencion de etiquetas elegida.

    Hay dos convenciones de COCO y no coinciden: 4 en las 80 clases contiguas
    de YOLO y 5 en los ids originales que usan los modelos de torchvision.
    Equivocarse no da ningun error, solo hace que el detector devuelva otra
    clase, asi que conviene fijarlo con un test.
    """
    from lanzarote_tracker.config import COCO_AIRPLANE_CLASS_ID

    detection = get_config().detection
    assert detection.coco_label_convention in COCO_AIRPLANE_CLASS_ID
    assert (
        detection.coco_airplane_class_id
        == COCO_AIRPLANE_CLASS_ID[detection.coco_label_convention]
    )


def test_coco_detector_rejects_an_inconsistent_class_id() -> None:
    """Una combinacion incoherente de convencion e indice se rechaza al construir."""
    from lanzarote_tracker.detection.factory import create_detector

    config = get_config()
    broken = replace(
        config.detection,
        backend="coco",
        coco_label_convention="paper91",
        coco_airplane_class_id=4,
    )
    with pytest.raises(ValueError, match="airplane"):
        create_detector(broken)


def test_configured_backends_are_available() -> None:
    """El backend elegido de cada modulo esta en su lista de disponibles."""
    config = get_config()
    assert config.detection.backend in config.detection.available_backends
    assert config.tracking.backend in config.tracking.available_backends
    assert config.classification.backend in config.classification.available_backends


# --- Camara, captura y dataset ---------------------------------------------
def test_camera_backend_matches_platform() -> None:
    """El backend de la webcam es CAP_DSHOW en Windows y CAP_ANY fuera."""
    import platform

    backend = get_config().camera.backend
    expected = "CAP_DSHOW" if platform.system() == "Windows" else "CAP_ANY"
    assert backend == expected


def test_detection_period_is_positive() -> None:
    """K, el periodo de deteccion, es mayor que 0."""
    assert get_config().tracking.detect_every_k_frames > 0


def test_split_ratios_sum_to_one() -> None:
    """Las proporciones de train, val y test suman 1."""
    ratios = get_config().classification.split_ratios
    assert len(ratios) == 3
    assert abs(sum(ratios) - 1.0) < 1e-9


def test_dataset_is_split_by_clip() -> None:
    """La division del dataset es por clips completos, nunca por fotogramas."""
    assert get_config().classification.split_by == "clip"


def test_airport_bbox_is_well_formed() -> None:
    """El recuadro geografico del aeropuerto es coherente y rodea a GCRR."""
    identification = get_config().identification
    lat_min, lat_max, lon_min, lon_max = identification.bbox
    assert lat_min < lat_max
    assert lon_min < lon_max
    # El aeropuerto de Lanzarote esta aproximadamente en 28.95 N, -13.60 E.
    assert lat_min < 28.95 < lat_max
    assert lon_min < -13.60 < lon_max


def test_card_fields_exclude_dynamic_data() -> None:
    """La ficha no muestra altura, velocidad ni posicion: solo datos fijos."""
    fields = get_config().identification.card_fields
    forbidden = {"altitude", "velocity", "speed", "latitude", "longitude", "heading"}
    assert not forbidden.intersection(fields)


def test_credentials_are_read_from_environment_variable_names() -> None:
    """La configuracion guarda NOMBRES de variables de entorno, no credenciales."""
    identification = get_config().identification
    assert identification.username_env_var == "OPENSKY_USERNAME"
    assert identification.password_env_var == "OPENSKY_PASSWORD"


# --- Depuracion -------------------------------------------------------------
def test_debug_is_disabled_by_default() -> None:
    """La depuracion viene apagada: no se escriben imagenes sin pedirlo."""
    assert get_config().debug.enabled is False


def test_with_debug_returns_a_copy() -> None:
    """with_debug no modifica la configuracion original."""
    config = get_config()
    enabled = with_debug(config, True)
    assert enabled.debug.enabled is True
    assert config.debug.enabled is False
    assert enabled is not config


def test_replace_allows_overriding_a_single_block() -> None:
    """dataclasses.replace permite variantes sin tocar el resto de bloques."""
    config = get_config()
    variant = replace(config, debug=DebugConfig(enabled=True, log_level="DEBUG"))
    assert variant.debug.log_level == "DEBUG"
    assert variant.security == config.security
