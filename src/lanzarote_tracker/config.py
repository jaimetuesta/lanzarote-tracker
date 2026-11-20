"""Configuracion centralizada del proyecto.

TODOS los parametros ajustables del sistema viven en este modulo, agrupados en
dataclasses por bloque funcional. Ningun otro fichero debe contener valores
magicos: si hace falta una constante nueva, se anade aqui.

Uso tipico::

    from lanzarote_tracker.config import get_config

    config = get_config()
    n = config.security.consecutive_frames

Las dataclasses son inmutables (``frozen=True``) para evitar que un modulo
modifique por accidente la configuracion de otro. Para probar valores distintos
se construye una copia con :func:`dataclasses.replace`.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Tuple

# Un rango HSV es (minimo, maximo) con cada extremo en (H, S, V).
# IMPORTANTE: en OpenCV el tono H va de 0 a 179 (no de 0 a 359).
HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]

# Raiz del repositorio: este fichero esta en <raiz>/src/lanzarote_tracker/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Valor maximo del canal de tono H en OpenCV.
MAX_HUE: int = 179


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathsConfig:
    """Rutas de trabajo del proyecto, todas relativas a la raiz del repositorio."""

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    clips: Path = PROJECT_ROOT / "data" / "clips"
    frames: Path = PROJECT_ROOT / "data" / "frames"
    annotations: Path = PROJECT_ROOT / "data" / "annotations"
    splits: Path = PROJECT_ROOT / "data" / "splits"
    models: Path = PROJECT_ROOT / "models"
    results: Path = PROJECT_ROOT / "results"
    calibration_images: Path = PROJECT_ROOT / "calibration" / "images"
    calibration_results: Path = PROJECT_ROOT / "calibration" / "results"
    cards: Path = PROJECT_ROOT / "assets" / "cards"


# ---------------------------------------------------------------------------
# Camara (webcam del usuario)
# ---------------------------------------------------------------------------
def _default_camera_backend() -> str:
    """Devuelve el backend de captura adecuado al sistema operativo.

    ``cv2.CAP_DSHOW`` (DirectShow) solo existe en Windows y alli evita el
    retardo de varios segundos al abrir la webcam. En Linux y macOS se usa
    ``cv2.CAP_ANY`` y OpenCV elige.

    Returns:
        Nombre de la constante de OpenCV, sin el prefijo ``cv2.``.
    """
    if platform.system() == "Windows":
        return "CAP_DSHOW"
    return "CAP_ANY"


@dataclass(frozen=True)
class CameraConfig:
    """Parametros de la webcam, usada en calibracion y en el bloque de seguridad."""

    index: int = 0
    #: Nombre de la constante de backend de OpenCV, p. ej. "CAP_DSHOW" o "CAP_ANY".
    #: Se resuelve a su valor numerico con `common.io_utils.open_camera`.
    backend: str = field(default_factory=_default_camera_backend)
    width: int = 1280
    height: int = 720
    #: FPS solicitados a la camara. 0 significa "no forzar".
    fps: int = 30
    #: Voltear horizontalmente la imagen (modo espejo) al mostrarla.
    mirror: bool = True
    window_name: str = "LanzaroteTracker"
    #: Teclas que cierran las ventanas interactivas: 'q' y ESC.
    quit_keys: Tuple[int, ...] = (ord("q"), 27)


# ---------------------------------------------------------------------------
# Calibracion (Practica 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationConfig:
    """Parametros del tablero de ajedrez y de la calibracion pinhole."""

    #: Esquinas internas del tablero (columnas, filas), no numero de cuadrados.
    pattern_size: Tuple[int, int] = (9, 6)
    #: Lado del cuadrado del tablero impreso, en milimetros.
    square_size_mm: float = 25.0
    #: Numero minimo de imagenes validas para dar la calibracion por buena.
    min_images: int = 15
    #: Criterios de parada de cv2.cornerSubPix: (max_iter, epsilon).
    subpix_max_iter: int = 30
    subpix_epsilon: float = 0.001
    #: Ventana de busqueda de cv2.cornerSubPix, en pixeles.
    subpix_window: Tuple[int, int] = (11, 11)
    #: RMS de reproyeccion por encima del cual se avisa de calibracion pobre.
    max_acceptable_rms: float = 1.0
    #: Nombres de los ficheros de salida dentro de `PathsConfig.calibration_results`.
    npz_filename: str = "camera_calibration.npz"
    json_filename: str = "camera_calibration.json"


# ---------------------------------------------------------------------------
# Sistema de seguridad (desbloqueo con secuencia de tarjetas)
# ---------------------------------------------------------------------------
#: Rangos HSV por color de tarjeta. El rojo necesita DOS rangos porque su tono
#: esta partido por el extremo del circulo cromatico (cerca de 0 y cerca de 179).
#: Son valores de partida: se afinan con `scripts/tune_hsv.py` sobre las
#: tarjetas impresas reales y con la iluminacion real.
DEFAULT_HSV_RANGES: Dict[str, Tuple[HsvRange, ...]] = {
    "red": (
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (MAX_HUE, 255, 255)),
    ),
    "green": (((36, 80, 70), (89, 255, 255)),),
    "blue": (((90, 80, 70), (130, 255, 255)),),
    "yellow": (((20, 100, 100), (35, 255, 255)),),
}


@dataclass(frozen=True)
class SecurityConfig:
    """Parametros del desbloqueo por secuencia de 4 tarjetas."""

    # --- Secuencia -----------------------------------------------------------
    #: Longitud de la secuencia de desbloqueo.
    sequence_length: int = 4
    #: Secuencia correcta, como pares (forma, color). Se comprueba ENTERA al
    #: final, nunca tarjeta a tarjeta, para no filtrar informacion parcial.
    #: TODO: moverla a variable de entorno cuando el sistema se use de verdad.
    expected_sequence: Tuple[Tuple[str, str], ...] = (
        ("circle", "red"),
        ("triangle", "green"),
        ("square", "blue"),
        ("pentagon", "yellow"),
    )

    # --- Validacion temporal -------------------------------------------------
    #: N: fotogramas consecutivos con la misma lectura para dar una tarjeta por leida.
    consecutive_frames: int = 15
    #: Fotogramas consecutivos SIN tarjeta que separan una lectura de la siguiente.
    gap_frames: int = 10
    #: Segundos maximos sin ver ninguna tarjeta antes de abortar la secuencia.
    max_seconds_without_card: float = 10.0
    #: Intentos fallidos consecutivos que disparan el bloqueo.
    max_failed_attempts: int = 3
    #: Duracion del bloqueo tras agotar los intentos, en segundos.
    lockout_seconds: float = 30.0

    # --- Segmentacion por color ----------------------------------------------
    hsv_ranges: Dict[str, Tuple[HsvRange, ...]] = field(
        default_factory=lambda: dict(DEFAULT_HSV_RANGES)
    )
    #: Tamano del elemento estructurante de apertura y cierre morfologicos.
    morph_kernel_size: Tuple[int, int] = (5, 5)
    morph_open_iterations: int = 2
    morph_close_iterations: int = 2
    #: Area minima del contorno, en pixeles, para considerarlo una tarjeta.
    min_contour_area: float = 3000.0

    # --- Clasificacion de forma ----------------------------------------------
    #: Factor epsilon de cv2.approxPolyDP como fraccion del perimetro.
    approx_poly_epsilon_ratio: float = 0.04
    #: Circularidad C = 4*pi*A / P^2 por encima de la cual la forma es un circulo.
    circularity_threshold: float = 0.80
    #: Numero de vertices de approxPolyDP por forma poligonal.
    vertices_by_shape: Dict[str, int] = field(
        default_factory=lambda: {"triangle": 3, "square": 4, "pentagon": 5}
    )
    #: Relacion de aspecto admisible para distinguir cuadrado de rectangulo.
    square_aspect_tolerance: float = 0.25
    #: Aplicar correccion de distorsion antes de segmentar. Requiere calibracion.
    undistort: bool = True


# ---------------------------------------------------------------------------
# Captura del stream
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureConfig:
    """Parametros de la grabacion de clips del directo y de la extraccion de fotogramas."""

    #: Directo de Flightradar24 del aeropuerto de Lanzarote en YouTube.
    #: Se puede sobreescribir con la variable de entorno LANZAROTE_STREAM_URL.
    stream_url: str = field(
        default_factory=lambda: os.environ.get(
            "LANZAROTE_STREAM_URL", "https://www.youtube.com/watch?v=4nC8r5neaow"
        )
    )
    #: Altura maxima del formato que pide yt-dlp.
    max_height: int = 1080
    #: Clientes de YouTube que usa yt-dlp. Los clientes tv/ios/android evitan el
    #: error "The page needs to be reloaded" del cliente web.
    ytdlp_player_clients: Tuple[str, ...] = ("tv", "ios", "android")
    #: Duracion por defecto de cada clip grabado, en segundos.
    clip_seconds: float = 60.0
    #: FourCC del codec de cv2.VideoWriter y extension del fichero.
    fourcc: str = "mp4v"
    clip_extension: str = ".mp4"
    #: FPS con los que se escribe el clip si el stream no informa de los suyos.
    fallback_fps: float = 25.0
    #: Sufijo del JSON de metadatos que acompana a cada clip.
    metadata_suffix: str = ".json"
    #: Formato de la marca de tiempo UTC de inicio del clip (ISO 8601).
    timestamp_format: str = "%Y%m%dT%H%M%SZ"
    #: Al extraer fotogramas, guardar uno de cada `frame_step`.
    frame_step: int = 5
    frame_extension: str = ".png"


# ---------------------------------------------------------------------------
# Deteccion del avion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectionConfig:
    """Parametros de la deteccion del avion y eleccion de backend.

    Backends disponibles:
      - ``classic``: segmentacion propia del avion frente a cielo y pista por
        color y morfologia. ES EL BACKEND POR DEFECTO.
      - ``coco``: detector preentrenado (clase ``airplane``). OPCIONAL y
        DESACTIVADO: su uso de deep learning esta PENDIENTE DE APROBACION DEL
        PROFESOR.
    """

    backend: str = "classic"
    #: Backends reconocidos por la fabrica.
    available_backends: Tuple[str, ...] = ("classic", "coco")

    # --- Backend classic -----------------------------------------------------
    #: Rango HSV del cielo, que se resta para aislar el avion.
    sky_hsv_range: HsvRange = ((90, 20, 120), (130, 255, 255))
    #: Area minima y maxima del contorno del avion, en pixeles.
    min_area: float = 500.0
    max_area: float = 200000.0
    morph_kernel_size: Tuple[int, int] = (5, 5)
    morph_open_iterations: int = 1
    morph_close_iterations: int = 3
    #: Relacion de aspecto admisible de la caja del avion (ancho/alto).
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 6.0

    # --- Backend coco (DESACTIVADO) ------------------------------------------
    #: Indice de la clase `airplane` en COCO.
    coco_airplane_class_id: int = 4
    coco_confidence_threshold: float = 0.5
    #: Ruta a los pesos del detector. Vacia porque el backend esta desactivado.
    coco_weights_path: str = ""


# ---------------------------------------------------------------------------
# Seguimiento
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackingConfig:
    """Parametros del seguimiento entre detecciones.

    La camara del aeropuerto gira y hace zoom, asi que NO se usa sustraccion de
    fondo: se detecta cada K fotogramas y entre detecciones se sigue al avion.

    Backends disponibles:
      - ``lk``: flujo optico de Lucas-Kanade sobre puntos Shi-Tomasi dentro de
        la caja (Practica 5).
      - ``meanshift``: meanShift sobre el histograma de tono (Practica 5).
    """

    backend: str = "lk"
    available_backends: Tuple[str, ...] = ("lk", "meanshift")
    #: K: cada cuantos fotogramas se vuelve a detectar en vez de seguir.
    detect_every_k_frames: int = 15
    #: Fotogramas seguidos sin seguimiento valido antes de forzar una deteccion.
    max_frames_lost: int = 10

    # --- Shi-Tomasi (cv2.goodFeaturesToTrack) --------------------------------
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: float = 7.0
    corner_block_size: int = 7

    # --- Lucas-Kanade (cv2.calcOpticalFlowPyrLK) -----------------------------
    lk_window_size: Tuple[int, int] = (15, 15)
    lk_max_level: int = 2
    lk_max_iter: int = 10
    lk_epsilon: float = 0.03
    #: Puntos validos minimos para dar el seguimiento por bueno.
    min_tracked_points: int = 6

    # --- meanShift -----------------------------------------------------------
    #: Numero de bins del histograma de tono.
    hist_bins: int = 16
    meanshift_max_iter: int = 10
    meanshift_epsilon: float = 1.0


# ---------------------------------------------------------------------------
# Clasificacion frente/espaldas -> llegada/salida
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationConfig:
    """Parametros de la clasificacion de orientacion y de la logica llegada/salida.

    Backends disponibles:
      - ``bow``: bolsa de palabras SIFT + K-means + SVM (Practica 5, apartado A).
        ES LA LINEA BASE.
      - ``cnn``: CNN propia. OPCIONAL y PENDIENTE DE APROBACION DEL PROFESOR.
    """

    backend: str = "bow"
    available_backends: Tuple[str, ...] = ("bow", "cnn")
    #: Etiquetas de orientacion: el avion se ve de frente (viene hacia camara) o
    #: de espaldas (se aleja de camara).
    class_names: Tuple[str, ...] = ("front", "back")

    # --- Backend bow ---------------------------------------------------------
    #: Numero de palabras visuales del vocabulario (clusters de K-means).
    vocabulary_size: int = 200
    #: Tamano al que se reescala el recorte del avion antes de extraer SIFT.
    patch_size: Tuple[int, int] = (128, 128)
    svm_kernel: str = "rbf"
    svm_c: float = 1.0
    svm_gamma: str = "scale"
    random_state: int = 42
    vocabulary_filename: str = "bow_vocabulary.npz"
    classifier_filename: str = "bow_svm.joblib"

    # --- Backend cnn (DESACTIVADO) -------------------------------------------
    cnn_input_size: Tuple[int, int] = (128, 128)
    cnn_weights_filename: str = "cnn_orientation.weights"

    # --- Division del dataset ------------------------------------------------
    #: La division SIEMPRE es por clips completos, nunca por fotogramas: dos
    #: fotogramas del mismo clip son casi identicos y contaminarian la validacion.
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_by: str = "clip"

    # --- Logica llegada/salida -----------------------------------------------
    #: Sentido de uso de la pista, que junto a frente/espaldas decide si el
    #: avion aterriza o despega. Pista 03/21 de GCRR.
    runway_in_use: str = "03"
    available_runways: Tuple[str, ...] = ("03", "21")
    #: Fotogramas coherentes antes de fijar el veredicto llegada/salida.
    decision_frames: int = 10


# ---------------------------------------------------------------------------
# Identificacion del vuelo (OpenSky Network)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IdentificationConfig:
    """Parametros del cliente de OpenSky y del emparejamiento avion-vuelo."""

    #: Codigos del aeropuerto de Lanzarote.
    airport_icao: str = "GCRR"
    airport_iata: str = "ACE"
    #: Recuadro geografico alrededor del aeropuerto: (lat_min, lat_max, lon_min, lon_max).
    bbox: Tuple[float, float, float, float] = (28.85, 29.05, -13.70, -13.50)
    base_url: str = "https://opensky-network.org/api"
    #: Nombres de las variables de entorno con las credenciales. NUNCA el valor.
    username_env_var: str = "OPENSKY_USERNAME"
    password_env_var: str = "OPENSKY_PASSWORD"
    request_timeout_seconds: float = 15.0
    #: Ventana temporal de las consultas de llegadas/salidas, en horas.
    query_window_hours: int = 2
    #: Campos fijos que se muestran en la ficha. NO se muestran altura,
    #: velocidad ni posicion: el enunciado solo admite datos fijos del avion.
    card_fields: Tuple[str, ...] = (
        "callsign",
        "registration",
        "aircraft_model",
        "origin",
        "destination",
    )


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisualizationConfig:
    """Parametros del dibujo de cajas, ficha de datos y FPS incrustado."""

    #: Colores BGR (OpenCV trabaja en BGR, no en RGB).
    box_color: Tuple[int, int, int] = (0, 255, 0)
    box_thickness: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)
    card_background_color: Tuple[int, int, int] = (0, 0, 0)
    #: Opacidad del fondo de la ficha, de 0 (transparente) a 1 (opaco).
    card_alpha: float = 0.6
    card_margin: int = 12
    card_width: int = 320
    card_line_height: int = 24
    font_scale: float = 0.6
    font_thickness: int = 1
    #: Esquina donde se dibuja el FPS: "top-left", "top-right", "bottom-left",
    #: "bottom-right".
    fps_position: str = "top-left"
    #: Fotogramas de la media movil del contador de FPS.
    fps_window: int = 30
    #: El video de entrega debe llevar el FPS incrustado.
    draw_fps: bool = True


# ---------------------------------------------------------------------------
# Depuracion: secuencia de transformacion para el informe
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DebugConfig:
    """Control del registro de imagenes intermedias de cada pipeline.

    El enunciado exige en el informe una figura con TODAS las transformaciones
    de la imagen. Con `enabled=True` cada pipeline guarda sus pasos intermedios
    en ``results/<modulo>/``.
    """

    enabled: bool = False
    output_root: Path = PROJECT_ROOT / "results"
    image_extension: str = ".png"
    #: Guardar tambien un mosaico con todos los pasos en una sola figura.
    save_mosaic: bool = True
    mosaic_columns: int = 3
    #: Altura a la que se reescala cada paso dentro del mosaico, en pixeles.
    mosaic_tile_height: int = 240
    #: Nivel de logging del proyecto: "DEBUG", "INFO", "WARNING", "ERROR".
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Configuracion global
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """Agrupa la configuracion de todos los bloques del sistema."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    identification: IdentificationConfig = field(default_factory=IdentificationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def get_config() -> AppConfig:
    """Devuelve la configuracion por defecto del proyecto.

    Returns:
        Una instancia nueva de :class:`AppConfig` con todos los valores por
        defecto. Para variantes, usar ``dataclasses.replace`` sobre el bloque
        que se quiera cambiar::

            config = get_config()
            config = replace(config, debug=replace(config.debug, enabled=True))
    """
    return AppConfig()


def with_debug(config: AppConfig, enabled: bool = True) -> AppConfig:
    """Devuelve una copia de la configuracion con la depuracion activada o no.

    Args:
        config: Configuracion de partida.
        enabled: Si se registran las imagenes intermedias de cada pipeline.

    Returns:
        Una copia de `config` con `debug.enabled` igual a `enabled`.
    """
    return replace(config, debug=replace(config.debug, enabled=enabled))


__all__ = [
    "MAX_HUE",
    "PROJECT_ROOT",
    "DEFAULT_HSV_RANGES",
    "HsvRange",
    "PathsConfig",
    "CameraConfig",
    "CalibrationConfig",
    "SecurityConfig",
    "CaptureConfig",
    "DetectionConfig",
    "TrackingConfig",
    "ClassificationConfig",
    "IdentificationConfig",
    "VisualizationConfig",
    "DebugConfig",
    "AppConfig",
    "get_config",
    "with_debug",
]
