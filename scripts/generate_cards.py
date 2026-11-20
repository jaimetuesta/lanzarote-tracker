"""Genera las 16 tarjetas imprimibles del desbloqueo: 4 formas x 4 colores.

Cada tarjeta es una imagen con fondo blanco y la figura rellena del color
correspondiente, centrada y con un margen amplio, para que el contorno se
segmente bien aunque la tarjeta se sostenga con la mano.

Los colores se eligen en los centros de los rangos HSV de `config.security`,
asi que las tarjetas generadas y los umbrales de segmentacion son coherentes
por construccion. Aun asi, conviene reajustar los rangos con
`scripts/tune_hsv.py` sobre las tarjetas YA IMPRESAS y con la iluminacion real,
porque la impresora y la luz cambian el tono.

Uso::

    python scripts/generate_cards.py
    python scripts/generate_cards.py --size 800 --output assets/cards
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from lanzarote_tracker.common.io_utils import write_image
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import MAX_HUE, AppConfig, get_config
from lanzarote_tracker.security.types import Color, Shape

_logger = get_logger(__name__)

Image = np.ndarray

#: Fraccion del lado de la tarjeta que ocupa la figura.
FIGURE_RATIO = 0.62


def representative_bgr(color: Color, config: AppConfig) -> Tuple[int, int, int]:
    """Devuelve el color BGR representativo de un color de tarjeta.

    Se toma el punto medio del PRIMER rango HSV configurado para ese color, con
    saturacion y valor altos, y se convierte a BGR. Para el rojo, cuyo tono esta
    partido por el extremo del circulo, el primer rango es el de tono cercano a
    0, que da un rojo correcto.

    Args:
        color: Color de la tarjeta.
        config: Configuracion de la aplicacion.

    Returns:
        El color en BGR, que es el orden que usa OpenCV.
    """
    (low, high) = config.security.hsv_ranges[color.value][0]
    hue = int(round((low[0] + high[0]) / 2.0)) % (MAX_HUE + 1)
    hsv = np.uint8([[[hue, 235, 235]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def regular_polygon(center: Tuple[int, int], radius: float, sides: int) -> np.ndarray:
    """Calcula los vertices de un poligono regular con un vertice hacia arriba.

    Args:
        center: Centro del poligono, en pixeles.
        radius: Radio de la circunferencia circunscrita, en pixeles.
        sides: Numero de lados.

    Returns:
        Un array (sides, 2) de enteros con los vertices, listo para cv2.fillPoly.
    """
    cx, cy = center
    # -pi/2 pone el primer vertice arriba, que es como se reconocen las figuras.
    angles = [(-math.pi / 2.0) + (2.0 * math.pi * index / sides) for index in range(sides)]
    points = [
        (int(round(cx + radius * math.cos(angle))), int(round(cy + radius * math.sin(angle))))
        for angle in angles
    ]
    return np.array(points, dtype=np.int32)


def draw_shape(
    canvas: Image,
    shape: Shape,
    color_bgr: Tuple[int, int, int],
) -> Image:
    """Dibuja la figura rellena en el centro de una COPIA del lienzo.

    El cuadrado se dibuja con sus lados paralelos a los bordes (no como un
    poligono regular girado), que es como se imprime y como se mostrara a la
    camara.

    Args:
        canvas: Lienzo cuadrado de fondo, que NO se modifica.
        shape: Forma a dibujar.
        color_bgr: Color de relleno en BGR.

    Returns:
        Una copia del lienzo con la figura dibujada.
    """
    card = canvas.copy()
    side = card.shape[0]
    center = (side // 2, side // 2)
    radius = (side * FIGURE_RATIO) / 2.0

    if shape is Shape.CIRCLE:
        cv2.circle(card, center, int(round(radius)), color_bgr, thickness=-1, lineType=cv2.LINE_AA)
    elif shape is Shape.SQUARE:
        half = int(round(radius * math.sqrt(2.0) / 2.0))
        top_left = (center[0] - half, center[1] - half)
        bottom_right = (center[0] + half, center[1] + half)
        cv2.rectangle(card, top_left, bottom_right, color_bgr, thickness=-1)
    elif shape is Shape.TRIANGLE:
        cv2.fillPoly(card, [regular_polygon(center, radius, 3)], color_bgr, lineType=cv2.LINE_AA)
    elif shape is Shape.PENTAGON:
        cv2.fillPoly(card, [regular_polygon(center, radius, 5)], color_bgr, lineType=cv2.LINE_AA)
    else:  # pragma: no cover - el enum no tiene mas valores
        raise ValueError(f"Forma desconocida: {shape}")
    return card


def build_card(
    shape: Shape,
    color: Color,
    size: int,
    config: AppConfig,
    label: bool = True,
) -> Image:
    """Construye la imagen de una tarjeta.

    Args:
        shape: Forma de la tarjeta.
        color: Color de la tarjeta.
        size: Lado de la tarjeta en pixeles.
        config: Configuracion de la aplicacion.
        label: Si se escribe el nombre de la tarjeta en una esquina. Ayuda a no
            confundirse al imprimirlas y recortarlas; el texto queda fuera de la
            figura y no estorba a la segmentacion.

    Returns:
        La tarjeta como imagen BGR.
    """
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    card = draw_shape(canvas, shape, representative_bgr(color, config))
    # Marco fino para poder recortar la tarjeta por su borde.
    cv2.rectangle(card, (0, 0), (size - 1, size - 1), (200, 200, 200), thickness=2)
    if label:
        cv2.putText(
            card,
            f"{shape.value} / {color.value}",
            (14, size - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            size / 1400.0,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
    return card


def generate_all(output_dir: Path, size: int, config: AppConfig, label: bool) -> List[Path]:
    """Genera las 16 tarjetas y las guarda en disco.

    Args:
        output_dir: Directorio de destino.
        size: Lado de cada tarjeta en pixeles.
        config: Configuracion de la aplicacion.
        label: Si se escribe el nombre de la tarjeta en una esquina.

    Returns:
        Las rutas de las tarjetas escritas.
    """
    written: List[Path] = []
    for shape in Shape:
        for color in Color:
            card = build_card(shape, color, size, config, label=label)
            path = output_dir / f"{shape.value}_{color.value}.png"
            written.append(write_image(path, card))
    return written


def build_sheet(cards: Dict[str, Image], columns: int = 4) -> Image:
    """Compone una hoja de contacto con todas las tarjetas, para imprimir de una vez.

    Args:
        cards: Diccionario de nombre a imagen de tarjeta, todas del mismo tamano.
        columns: Numero de columnas de la hoja.

    Returns:
        La hoja completa como imagen BGR.
    """
    tiles = list(cards.values())
    rows = [np.hstack(tiles[start : start + columns]) for start in range(0, len(tiles), columns)]
    return np.vstack(rows)


def main() -> int:
    """Punto de entrada del script.

    Returns:
        0 si se generan las tarjetas.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Genera las 16 tarjetas imprimibles del desbloqueo (4 formas x 4 "
            "colores) usando los colores centrales de los rangos HSV de la "
            "configuracion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config = get_config()
    parser.add_argument(
        "--output",
        type=Path,
        default=config.paths.cards,
        help="Directorio donde se guardan las tarjetas.",
    )
    parser.add_argument(
        "--size", type=int, default=700, help="Lado de cada tarjeta, en pixeles."
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="No escribir el nombre de la tarjeta en la esquina.",
    )
    parser.add_argument(
        "--sheet",
        action="store_true",
        help="Generar ademas una hoja de contacto con las 16 tarjetas.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    written = generate_all(args.output, args.size, config, label=not args.no_label)
    _logger.info("Generadas %d tarjetas en %s", len(written), args.output)

    if args.sheet:
        cards = {path.stem: cv2.imread(str(path)) for path in written}
        sheet_path = write_image(args.output / "sheet.png", build_sheet(cards))
        _logger.info("Hoja de contacto en %s", sheet_path)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
