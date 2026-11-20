"""Deteccion del tablero de ajedrez en las fotos de calibracion.

Practica 2: cv2.findChessboardCorners + cv2.cornerSubPix.

IMPORTANTE sobre `CalibrationConfig.pattern_size`: son las **esquinas internas**
del tablero, no el numero de cuadrados. Un tablero de 10x7 cuadrados tiene 9x6
esquinas internas. Si el valor no coincide con el tablero real,
cv2.findChessboardCorners no encuentra NADA en ninguna foto: es un fallo de
todo o nada, no una deteccion peor. Para salir de dudas esta
:func:`detect_pattern_size`, que prueba los tamanos habituales sobre una foto.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import CalibrationConfig

Image = np.ndarray
#: Esquinas detectadas en una imagen, con forma (N, 1, 2) y dtype float32.
Corners = np.ndarray

_logger = get_logger(__name__)

#: Banderas de cv2.findChessboardCorners. Se omite CALIB_CB_FAST_CHECK a
#: proposito: acelera el descarte de imagenes sin tablero, pero tambien
#: descarta tableros validos que aparecen girados o poco contrastados, y aqui
#: interesa aprovechar todas las fotos.
FIND_FLAGS: int = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def to_gray(image: Image) -> Image:
    """Devuelve la imagen en escala de grises, que es lo que esperan las funciones de OpenCV.

    Args:
        image: Imagen BGR o ya en escala de grises.

    Returns:
        La imagen en escala de grises. Si ya lo estaba, se devuelve tal cual.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def parse_pattern_size(text: str) -> Tuple[int, int]:
    """Interpreta un tamano de patron escrito como texto, p. ej. "9x6".

    Acepta "9x6", "9X6", "9,6" y "9 6".

    Args:
        text: Tamano del patron en esquinas internas (columnas x filas).

    Returns:
        El par (columnas, filas).

    Raises:
        ValueError: Si el texto no tiene el formato esperado o algun valor es
            menor que 2, que es el minimo para que haya un tablero.
    """
    parts = re.split(r"[x,X\s]+", text.strip())
    if len(parts) != 2:
        raise ValueError(
            f"Tamano de patron invalido: {text!r}. Escribelo como columnas x filas, "
            "en esquinas internas, por ejemplo '9x6'."
        )
    try:
        columns, rows = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise ValueError(
            f"Tamano de patron invalido: {text!r}. Los dos valores deben ser enteros."
        ) from error
    if columns < 2 or rows < 2:
        raise ValueError(
            f"Tamano de patron invalido: {text!r}. Ambos valores deben ser 2 o mas."
        )
    return columns, rows


def find_corners(
    image: Image,
    config: CalibrationConfig,
    refine: bool = True,
) -> Optional[Corners]:
    """Busca las esquinas internas del tablero en una imagen.

    Practica 2: cv2.findChessboardCorners sobre la imagen en escala de grises y,
    si se encuentran, refinado subpixel con cv2.cornerSubPix usando la ventana y
    los criterios de parada de la configuracion.

    Args:
        image: Imagen BGR o en escala de grises con el tablero.
        config: Configuracion de calibracion, de la que se toman `pattern_size`
            y los parametros del refinado subpixel.
        refine: Si se aplica el refinado subpixel.

    Returns:
        Las esquinas con forma (filas*columnas, 1, 2) y dtype float32, o None
        si el tablero no aparece completo en la imagen.
    """
    gray = to_gray(image)
    found, corners = cv2.findChessboardCorners(gray, config.pattern_size, FIND_FLAGS)
    if not found or corners is None:
        return None
    if refine:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            config.subpix_max_iter,
            config.subpix_epsilon,
        )
        corners = cv2.cornerSubPix(
            gray, corners, config.subpix_window, (-1, -1), criteria
        )
    return corners


def build_object_points(config: CalibrationConfig) -> np.ndarray:
    """Construye las coordenadas 3D teoricas de las esquinas del tablero.

    El tablero se supone plano con Z = 0, y las esquinas se separan
    `config.square_size_mm` milimetros entre si, de modo que la calibracion
    devuelva magnitudes en milimetros.

    NOTA: el lado del cuadrado solo fija la ESCALA de la escena, asi que afecta
    a los vectores de traslacion pero NO a la matriz K ni a los coeficientes de
    distorsion. Para corregir la distorsion, que es el unico uso que le da este
    proyecto a la calibracion, un error de escala es irrelevante.

    Args:
        config: Configuracion de calibracion, de la que se toman `pattern_size`
            y `square_size_mm`.

    Returns:
        Un array (filas*columnas, 3) de dtype float32 con las coordenadas del
        patron, identico para todas las imagenes.
    """
    columns, rows = config.pattern_size
    points = np.zeros((rows * columns, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points * float(config.square_size_mm)


def collect_corners(
    images: Sequence[Image],
    config: CalibrationConfig,
    recorder: Optional[DebugRecorder] = None,
) -> Tuple[List[np.ndarray], List[Corners], Tuple[int, int]]:
    """Recorre las imagenes de calibracion y recoge las correspondencias 3D-2D.

    Practica 2: por cada imagen valida se anade una copia de los puntos 3D del
    patron y las esquinas 2D detectadas. Las imagenes donde no aparece el
    tablero completo se descartan.

    Args:
        images: Fotos del tablero tomadas con la webcam.
        config: Configuracion de calibracion.
        recorder: Registro opcional de la secuencia de transformacion, al que se
            entrega cada imagen con sus esquinas dibujadas
            (cv2.drawChessboardCorners).

    Returns:
        Una tupla (puntos 3D, puntos 2D, tamano de imagen en (ancho, alto)).

    Raises:
        ValueError: Si no hay imagenes, si no todas tienen el mismo tamano o si
            no se alcanza `config.min_images` imagenes validas.
    """
    if not images:
        raise ValueError("No hay imagenes de calibracion que procesar.")

    first_height, first_width = images[0].shape[:2]
    image_size = (first_width, first_height)

    pattern_points = build_object_points(config)
    object_points: List[np.ndarray] = []
    image_points: List[Corners] = []

    for index, image in enumerate(images):
        height, width = image.shape[:2]
        if (width, height) != image_size:
            raise ValueError(
                "Todas las fotos de calibracion deben tener el mismo tamano. "
                f"La primera es {image_size[0]}x{image_size[1]} y la numero "
                f"{index} es {width}x{height}. No mezcles resoluciones ni "
                "recortes las fotos despues de tomarlas."
            )

        corners = find_corners(image, config)
        if corners is None:
            _logger.debug("Tablero no encontrado en la imagen %d; se descarta.", index)
            continue

        object_points.append(pattern_points.copy())
        image_points.append(corners)

        if recorder is not None and recorder.enabled:
            drawn = image.copy()
            cv2.drawChessboardCorners(drawn, config.pattern_size, corners, True)
            recorder.record(f"corners_{index:03d}", drawn)

    if len(object_points) < config.min_images:
        columns, rows = config.pattern_size
        raise ValueError(
            f"Solo se ha encontrado el tablero en {len(object_points)} de "
            f"{len(images)} fotos, y la calibracion necesita al menos "
            f"{config.min_images}.\n"
            f"Se buscaba un patron de {columns}x{rows} ESQUINAS INTERNAS "
            "(cuadrados menos uno por lado). Si no coincide con tu tablero, "
            "no se encuentra en ninguna foto: comprueba el tamano con "
            "`python scripts/calibrate_camera.py --detect-pattern`."
        )

    _logger.info(
        "Tablero encontrado en %d de %d fotos.", len(object_points), len(images)
    )
    return object_points, image_points, image_size


#: Ancho al que se reduce la foto antes de probar tamanos de patron. La
#: busqueda prueba decenas de combinaciones y cv2.findChessboardCorners es caro
#: a resolucion completa: a 640 px tarda del orden de cuatro veces menos y
#: sigue reconociendo el tablero, porque aqui solo interesa SABER el tamano, no
#: localizar las esquinas con precision.
PROBE_WIDTH: int = 640


def detect_pattern_size(
    image: Image,
    max_size: int = 12,
    min_size: int = 3,
    first_only: bool = True,
    probe_width: int = PROBE_WIDTH,
) -> List[Tuple[int, int]]:
    """Prueba tamanos de patron sobre una foto y devuelve los que encajan.

    Sirve para no tener que contar los cuadrados del tablero a mano: se ejecuta
    una vez sobre una foto en la que el tablero se vea entero y de frente.

    Se prueban los tamanos de mayor a menor numero de esquinas y, por defecto,
    se para en el primero que encaja: como el tablero real es el de mas
    esquinas, el primer acierto ya es la respuesta y no hace falta agotar las
    decenas de combinaciones restantes.

    No usa el refinado subpixel, que aqui no aporta nada y solo ralentiza la
    busqueda.

    Args:
        image: Foto del tablero, a ser posible completo y bien iluminado.
        max_size: Numero maximo de esquinas internas por lado que se prueba.
        min_size: Numero minimo de esquinas internas por lado que se prueba.
        first_only: Si se para en el primer tamano que encaja. Con False se
            prueban todos, que es mucho mas lento.
        probe_width: Ancho al que se reduce la foto antes de probar. Con 0 se
            usa la resolucion original.

    Returns:
        Los tamanos (columnas, filas) que han encajado, de mayor a menor numero
        de esquinas. Con `first_only` la lista trae como mucho un elemento.
    """
    gray = to_gray(image)
    if probe_width and gray.shape[1] > probe_width:
        scale = probe_width / float(gray.shape[1])
        gray = cv2.resize(
            gray,
            (probe_width, max(int(round(gray.shape[0] * scale)), 1)),
            interpolation=cv2.INTER_AREA,
        )

    candidates = sorted(
        (
            (columns, rows)
            for columns in range(min_size, max_size + 1)
            for rows in range(min_size, columns + 1)
        ),
        key=lambda size: (size[0] * size[1], size[0]),
        reverse=True,
    )

    matches: List[Tuple[int, int]] = []
    for size in candidates:
        found, _ = cv2.findChessboardCorners(gray, size, FIND_FLAGS)
        if found:
            matches.append(size)
            if first_only:
                break
    return matches


__all__ = [
    "FIND_FLAGS",
    "PROBE_WIDTH",
    "Corners",
    "to_gray",
    "parse_pattern_size",
    "find_corners",
    "build_object_points",
    "collect_corners",
    "detect_pattern_size",
]
