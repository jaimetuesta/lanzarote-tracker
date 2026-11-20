"""Ajuste en vivo de los rangos HSV con trackbars.

Adaptado del `get_hsv_color_ranges` de la Practica 1, con dos cambios:

- el tono H va de 0 a 179, que es el rango real de OpenCV (la practica usaba
  valores de 0 a 255 en algunos apartados y eso descuadra los umbrales);
- trabaja sobre la webcam en vivo, no sobre una imagen fija, para poder mover la
  tarjeta y ver como aguanta el umbral con los cambios de luz.

Al salir imprime los rangos elegidos en el formato exacto de
`SecurityConfig.hsv_ranges`, listos para pegarlos en `config.py`.

RECORDATORIO sobre el rojo: su tono esta partido por el extremo del circulo
cromatico, asi que NO se puede capturar con un solo rango. Este script ajusta
un rango cada vez; para el rojo hay que ajustar dos (uno con H cerca de 0 y otro
con H cerca de 179) y unirlos en la configuracion.

Uso::

    python scripts/tune_hsv.py
    python scripts/tune_hsv.py --image assets/cards/circle_red.png
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from lanzarote_tracker.common.io_utils import is_quit_key, load_image, open_camera
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import MAX_HUE, AppConfig, get_config

_logger = get_logger(__name__)

Image = np.ndarray

WINDOW_NAME = "Ajuste HSV - 'q' o ESC para salir"
#: Nombre de cada trackbar y su valor maximo. H llega a 179, S y V a 255.
TRACKBARS: Tuple[Tuple[str, int], ...] = (
    ("H min", MAX_HUE),
    ("H max", MAX_HUE),
    ("S min", 255),
    ("S max", 255),
    ("V min", 255),
    ("V max", 255),
)


def _noop(value: int) -> None:
    """Callback vacio que exige cv2.createTrackbar.

    Args:
        value: Valor nuevo del trackbar, que no se usa.
    """
    return None


def create_window(initial: Tuple[int, int, int, int, int, int]) -> None:
    """Crea la ventana con los seis trackbars.

    Args:
        initial: Valores iniciales en el orden (H min, H max, S min, S max,
            V min, V max).
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 900, 700)
    for (name, maximum), value in zip(TRACKBARS, initial):
        cv2.createTrackbar(name, WINDOW_NAME, int(value), maximum, _noop)


def read_trackbars() -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Lee el rango HSV actual de los trackbars.

    Returns:
        El par ((H,S,V) minimo, (H,S,V) maximo).
    """
    values = [cv2.getTrackbarPos(name, WINDOW_NAME) for name, _ in TRACKBARS]
    low = (values[0], values[2], values[4])
    high = (values[1], values[3], values[5])
    return low, high


def build_view(frame: Image, low: Tuple[int, int, int], high: Tuple[int, int, int]) -> Image:
    """Compone la vista con el fotograma, la mascara y el resultado enmascarado.

    Args:
        frame: Fotograma BGR.
        low: Extremo inferior del rango HSV.
        high: Extremo superior del rango HSV.

    Returns:
        Una imagen con las tres vistas apiladas horizontalmente.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([frame, mask_bgr, masked])


def print_range(low: Tuple[int, int, int], high: Tuple[int, int, int]) -> None:
    """Imprime el rango en el formato de `SecurityConfig.hsv_ranges`.

    Args:
        low: Extremo inferior del rango HSV.
        high: Extremo superior del rango HSV.
    """
    print()
    print("Rango elegido, listo para pegar en config.DEFAULT_HSV_RANGES:")
    print(f'    "<color>": ((({low[0]}, {low[1]}, {low[2]}), ({high[0]}, {high[1]}, {high[2]})),),')
    print()
    print("Para el ROJO hacen falta DOS rangos (H cerca de 0 y H cerca de 179):")
    print('    "red": (((0, S, V), (10, 255, 255)), ((170, S, V), (179, 255, 255))),')
    print()


def run(config: AppConfig, image_path: Optional[Path]) -> int:
    """Ejecuta el ajuste, sobre la webcam o sobre una imagen fija.

    Args:
        config: Configuracion de la aplicacion.
        image_path: Imagen sobre la que ajustar. Si es None se usa la webcam.

    Returns:
        0 si termina correctamente, 1 si no hay fuente de imagen.
    """
    static_frame: Optional[Image] = None
    capture = None

    if image_path is not None:
        try:
            static_frame = load_image(image_path)
        except FileNotFoundError as error:
            _logger.error("%s", error)
            return 1
    else:
        try:
            capture = open_camera(config.camera)
        except RuntimeError as error:
            _logger.error("%s", error)
            return 1

    create_window((0, MAX_HUE, 0, 255, 0, 255))
    _logger.info("Mueve los trackbars. Sal con 'q' o ESC para imprimir el rango.")

    low, high = (0, 0, 0), (MAX_HUE, 255, 255)
    try:
        while True:
            if static_frame is not None:
                frame = static_frame.copy()
            else:
                ok, frame = capture.read()  # type: ignore[union-attr]
                if not ok:
                    _logger.warning("Fotograma perdido; se cierra la camara.")
                    break
                if config.camera.mirror:
                    frame = cv2.flip(frame, 1)

            low, high = read_trackbars()
            cv2.imshow(WINDOW_NAME, build_view(frame, low, high))

            if is_quit_key(cv2.waitKey(30) & 0xFF, config.camera.quit_keys):
                break
    except KeyboardInterrupt:
        _logger.info("Interrumpido por el usuario.")
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()

    print_range(low, high)
    return 0


def main() -> int:
    """Punto de entrada del script.

    Returns:
        El codigo de salida del proceso.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ventana con trackbars para ajustar un rango HSV en vivo sobre la "
            "webcam o sobre una imagen. Adaptado de la Practica 1, con H de 0 a "
            "179. Al salir imprime el rango listo para pegar en config.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "El rojo necesita DOS rangos unidos, uno con H cerca de 0 y otro con "
            "H cerca de 179."
        ),
    )
    config = get_config()
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Ajustar sobre esta imagen en vez de sobre la webcam.",
    )
    parser.add_argument(
        "--camera", type=int, default=config.camera.index, help="Indice de la webcam."
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(config, camera=replace(config.camera, index=args.camera))
    return run(config, args.image)


if __name__ == "__main__":
    raise SystemExit(main())
