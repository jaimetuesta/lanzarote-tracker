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
from typing import Dict, Optional, Tuple

# Un rango HSV es (minimo, maximo) con cada extremo en (H, S, V).
# IMPORTANTE: en OpenCV el tono H va de 0 a 179 (no de 0 a 359).
HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]

# Raiz del repositorio: este fichero esta en <raiz>/src/lanzarote_tracker/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Valor maximo del canal de tono H en OpenCV.
MAX_HUE: int = 179

#: Indice de la clase `airplane` en cada convencion de etiquetas de COCO.
#: Ver `DetectionConfig.coco_label_convention`.
COCO_AIRPLANE_CLASS_ID: Dict[str, int] = {"contiguous80": 4, "paper91": 5}


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
    #: Datos de referencia que SI se versionan (base de aeropuertos).
    reference_data: Path = PROJECT_ROOT / "assets" / "data"
    #: Fichas de vuelo registradas durante el seguimiento en vivo.
    flight_records: Path = PROJECT_ROOT / "data" / "flights"


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
    #: Medido sobre el tablero real del proyecto con
    #: `python scripts/calibrate_camera.py --detect-pattern`.
    pattern_size: Tuple[int, int] = (8, 6)
    #: Lado del cuadrado del tablero impreso, en milimetros.
    #: Solo fija la escala de la escena: afecta a los vectores de traslacion,
    #: pero NO a la matriz K ni a los coeficientes de distorsion, que es lo
    #: unico que usa este proyecto.
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
#:
#: MEDIDOS el 27/11/2026 con `scripts/tune_hsv.py` sobre las tarjetas mostradas
#: EN LA PANTALLA DE UN IPAD, no impresas. Una pantalla reproduce casi exactos
#: los colores que `generate_cards.py` genera, asi que este es el caso FACIL:
#: hay que volver a medirlos sobre las tarjetas impresas antes de dar por buena
#: la robustez del sistema, y contar las dos medidas en el informe.
DEFAULT_HSV_RANGES: Dict[str, Tuple[HsvRange, ...]] = {
    # TODO: el segundo tramo del rojo esta a medias. Al medirlo, la ventana
    # solo mostraba el tramo (170..179) y el otro se perdio con el fallo de
    # impresion; el 10 de aqui es provisional y hay que volver a medir el rojo.
    "red": (
        ((0, 170, 133), (10, 255, 255)),
        ((170, 170, 133), (MAX_HUE, 255, 255)),
    ),
    "green": (((60, 190, 137), (77, 255, 255)),),
    "blue": (((98, 175, 192), (114, 255, 255)),),
    "yellow": (((19, 129, 98), (35, 255, 255)),),
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
    #: FPS con los que se escribe el clip si no se pueden medir.
    fallback_fps: float = 25.0

    #: Sufijo del JSON de metadatos que acompana a cada clip.
    metadata_suffix: str = ".json"
    #: Formato de la marca de tiempo UTC de inicio del clip (ISO 8601).
    timestamp_format: str = "%Y%m%dT%H%M%SZ"
    # --- Disparo por trafico real --------------------------------------------
    #: Grabar a ciegas desperdicia disco y tiempo: fuera de las horas punta la
    #: pista esta vacia casi siempre. Con `--when-active` se vigila ADS-B y solo
    #: se graba cuando hay un avion operando de verdad.
    #: Radio de vigilancia, en millas nauticas. Mayor que el de la pista para
    #: tener margen: un avion en aproximacion entra aqui antes de tomar tierra.
    trigger_radius_nm: int = 12
    #: Altura por debajo de la cual un avion dentro del radio se considera en
    #: aproximacion o despegando, no de paso.
    trigger_altitude_ft: int = 6000
    #: Segundos entre consultas a ADS-B mientras se espera.
    trigger_poll_seconds: int = 15
    #: Segundos de espera tras grabar antes de poder volver a disparar, para no
    #: encadenar clips del mismo avion.
    trigger_cooldown_seconds: int = 45

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
        color y morfologia. ES EL BACKEND POR DEFECTO y la linea principal.
      - ``coco``: detector preentrenado (clase ``airplane``). OPCIONAL. El
        profesor ha aprobado el uso de deep learning, pero el proyecto sigue
        siendo de vision por ordenador tradicional: este backend es una
        alternativa con la que comparar en el informe, no el camino principal.
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

    # --- Backend coco (OPCIONAL) ---------------------------------------------
    #: Convencion de indices de clase de COCO que usa el detector elegido. Hay
    #: DOS y NO coinciden:
    #:   - "contiguous80": las 80 clases contiguas con las que trabajan YOLO y
    #:     ultralytics. Ahi `airplane` es la 4.
    #:   - "paper91": los ids originales del dataset COCO, que son los que
    #:     devuelven los modelos preentrenados de torchvision. Ahi `airplane`
    #:     es la 5.
    #: Confundirlas NO da ningun error: el detector devuelve otra clase
    #: (motocicletas en vez de aviones) y el fallo se manifiesta como "el
    #: detector no funciona". Al elegir modelo hay que fijar esto primero.
    coco_label_convention: str = "contiguous80"
    #: Indice de la clase `airplane` segun `coco_label_convention`. El test
    #: `test_coco_class_id_matches_its_convention` comprueba que son coherentes.
    coco_airplane_class_id: int = 4
    coco_confidence_threshold: float = 0.5
    #: Ruta a los pesos del detector. Vacia hasta elegir modelo.
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
        ES LA LINEA BASE y el backend por defecto.
      - ``cnn``: CNN propia. OPCIONAL. El profesor ha aprobado el uso de deep
        learning, pero el proyecto sigue siendo de vision por ordenador
        tradicional: la CNN es la alternativa con la que comparar en el
        informe, y `bow` sigue siendo la linea base.
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

    # --- Backend cnn (OPCIONAL) ----------------------------------------------
    cnn_input_size: Tuple[int, int] = (128, 128)
    cnn_weights_filename: str = "cnn_orientation.weights"
    cnn_batch_size: int = 32
    cnn_epochs: int = 20
    cnn_learning_rate: float = 1e-3

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
    """Parametros de la identificacion del vuelo.

    FUENTES DE DATOS. Hay dos caminos y el primario NO es OpenSky:

    - **ADS-B (primario)**: `adsb.lol` y `adsb.fi` dan los aviones que hay ahora
      mismo alrededor del aeropuerto, y `adsbdb.com` da modelo, matricula,
      aerolinea y ruta. No necesitan credenciales y responden AL MOMENTO.
    - **OpenSky (secundario)**: sus consultas de llegadas y salidas por
      aeropuerto solo estan disponibles AL DIA SIGUIENTE, asi que no sirven
      para rellenar la ficha mientras se graba. Se conserva porque permite
      confirmar a posteriori lo deducido y porque su cobertura es distinta.
    """

    #: Codigos del aeropuerto de Lanzarote.
    airport_icao: str = "GCRR"
    airport_iata: str = "ACE"
    #: Nombre que se muestra en la ficha cuando el otro extremo de la ruta es ACE.
    airport_label: str = "ACE (Lanzarote Island)"
    #: Centro de la busqueda de aviones: cabecera del aeropuerto.
    airport_lat: float = 28.945
    airport_lon: float = -13.605
    #: Recuadro geografico para OpenSky: (lat_min, lat_max, lon_min, lon_max).
    bbox: Tuple[float, float, float, float] = (28.85, 29.05, -13.70, -13.50)

    # --- Fuente primaria: ADS-B ----------------------------------------------
    #: Fuentes de aviones en vivo, en orden. Se prueba la siguiente si una falla.
    adsb_sources: Tuple[str, ...] = (
        "https://api.adsb.lol/v2",
        "https://opendata.adsb.fi/api/v2",
    )
    #: Base de datos de aeronaves y rutas.
    adsbdb_url: str = "https://api.adsbdb.com/v0"
    #: Radio de busqueda alrededor del aeropuerto, en millas nauticas.
    search_radius_nm: int = 3
    request_timeout_seconds: float = 15.0

    # --- Fuente secundaria: OpenSky ------------------------------------------
    opensky_base_url: str = "https://opensky-network.org/api"
    #: Endpoint del token OAuth2. OpenSky usa `client_credentials`: NO es
    #: usuario y contrasena.
    opensky_token_url: str = (
        "https://auth.opensky-network.org/auth/realms/opensky-network"
        "/protocol/openid-connect/token"
    )
    #: Nombres de las variables de entorno con las credenciales. NUNCA el valor.
    client_id_env_var: str = "OPENSKY_CLIENT_ID"
    client_secret_env_var: str = "OPENSKY_CLIENT_SECRET"
    #: Margen de seguridad con el que se renueva el token antes de caducar.
    token_refresh_margin_seconds: float = 60.0

    # --- Deteccion de movimiento en la pista ---------------------------------
    #: Altura por encima de la cual se considera un vuelo de paso, en pies.
    max_altitude_ft: int = 5000
    #: Altura por debajo de la cual se considera que el avion esta en tierra.
    #: Hace falta porque no todos los aviones emiten el indicador "ground".
    ground_altitude_ft: int = 100
    #: Velocidad respecto al suelo minima, en nudos, para no considerar que un
    #: avion en tierra esta aparcado o rodando despacio.
    min_ground_speed_kt: int = 30
    #: Distancia maxima al eje de la pista para contar como "en pista", en metros.
    runway_margin_m: int = 75
    #: Velocidad vertical minima, en pies por minuto, para deducir si sube o baja.
    min_vertical_rate_fpm: int = 300
    #: Extremos de la pista 03/21 de GCRR, como ((lat, lon), (lat, lon)).
    #: TODO: rellenar con los umbrales reales de la pista (estan en runways.csv
    #: de OurAirports, que no venia en el material de Jaime). Mientras sea None,
    #: un avion en tierra solo cuenta como "en pista" si va a mas de
    #: `min_ground_speed_kt`, que es mas impreciso.
    runway_endpoints: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    # --- Ritmo del seguimiento en vivo ---------------------------------------
    #: Segundos entre consultas de la zona del aeropuerto.
    poll_seconds: int = 10
    #: Segundos entre consultas de un avion que ya ha despegado.
    follow_poll_seconds: int = 60
    #: Segundos tras los cuales se deja de seguir una salida.
    max_follow_seconds: int = 4 * 3600
    #: Segundos sin ver un avion para empezar su historial de cero.
    history_reset_seconds: int = 1800
    #: Segundos sin ver un avion para darlo por perdido.
    lost_after_seconds: int = 60
    #: Si se pierde por debajo de esta altura bajando, se da por aterrizado.
    lost_below_ft: int = 1000
    #: Altura por debajo de la cual se supone que una salida perdida ha
    #: aterrizado cerca de su ultima posicion conocida.
    landed_below_ft: int = 3000

    # --- Ficha ---------------------------------------------------------------
    #: Campos FIJOS que se muestran en la ficha. NO se muestran altura,
    #: velocidad ni posicion: el enunciado solo admite datos fijos del avion.
    card_fields: Tuple[str, ...] = (
        "movement",
        "callsign",
        "airline",
        "origin",
        "destination",
        "event_time",
        "aircraft_model",
        "registration",
    )
    #: Etiqueta en espanol de cada campo de la ficha, para mostrarla.
    card_labels: Dict[str, str] = field(
        default_factory=lambda: {
            "movement": "Llegada o salida",
            "callsign": "Indicativo",
            "airline": "Aerolinea",
            "origin": "Origen",
            "destination": "Destino",
            "event_time": "Hora",
            "aircraft_model": "Modelo",
            "registration": "Matricula",
        }
    )
    #: Texto con el que se rellena un campo desconocido.
    unknown_value: str = "-"
    #: Formato de la hora del evento que se muestra en la ficha.
    event_time_format: str = "%d/%m/%Y %H:%M UTC"


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
    "COCO_AIRPLANE_CLASS_ID",
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
