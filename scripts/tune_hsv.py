"""Ajuste de los rangos HSV de las tarjetas, con clic o con deslizadores.

COMO SE USA, en corto:

1. Pon una tarjeta delante de la camara.
2. **Haz clic encima de la figura de color.** El rango se calcula solo a partir
   de los pixeles que hay bajo el cursor.
3. Mira la vista del medio: tiene que quedar BLANCA sobre la figura y NEGRA en
   todo lo demas. Si no, haz clic en otro punto o retoca los deslizadores.
4. Sal con 'q' o ESC. Se imprime la linea lista para pegar en `config.py`.
5. Repite con cada color. El ROJO se resuelve solo: al hacer clic se detecta
   que cruza la costura del circulo cromatico y se imprimen sus DOS rangos.

Adaptado del `get_hsv_color_ranges` de la Practica 1, con dos cambios: el tono
va de 0 a 179, que es el rango real de OpenCV, y se mide la muestra en vez de
ajustar a ojo, que es mas rapido y se puede justificar en el informe.

Uso::

    python scripts/tune_hsv.py --color red
    python scripts/tune_hsv.py --image assets/cards/circle_red.png --color red
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from lanzarote_tracker.common.io_utils import is_quit_key, load_image, open_camera
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import MAX_HUE, AppConfig, HsvRange, get_config
from lanzarote_tracker.security.hsv_tuning import format_ranges, sample_hsv_range

_logger = get_logger(__name__)

Image = np.ndarray

WINDOW_NAME = "Ajuste HSV - clic sobre la figura, 'q' o ESC para salir"
#: Ancho de cada una de las tres vistas, en pixeles. Fijo para que las
#: coordenadas del raton se puedan convertir a las del fotograma.
PANE_WIDTH = 460
#: Lado del cuadrado que se muestrea alrededor del clic.
SAMPLE_PATCH = 15

TRACKBARS: Tuple[Tuple[str, int], ...] = (
    ("H min", MAX_HUE),
    ("H max", MAX_HUE),
    ("S min", 255),
    ("S max", 255),
    ("V min", 255),
    ("V max", 255),
)

#: Segundo rango, cuando el color cruza la costura (el rojo). Los deslizadores
#: solo pueden representar uno, asi que el otro se guarda aqui y se usa tanto
#: para la mascara que se ve como para lo que se imprime al salir.
_extra_range: Optional[HsvRange] = None
#: Ultimo fotograma, para poder muestrearlo desde el callback del raton.
_last_frame: Optional[Image] = None
#: Escala entre la vista mostrada y el fotograma original.
_scale: float = 1.0
#: Color que se esta ajustando, para poder imprimirlo ya al hacer clic.
_color: str = "red"


def _noop(value: int) -> None:
    """Callback vacio que exige cv2.createTrackbar.

    Args:
        value: Valor nuevo del trackbar, que no se usa.
    """
    return None


def set_trackbars(low: Tuple[int, int, int], high: Tuple[int, int, int]) -> None:
    """Coloca los deslizadores en un rango concreto.

    Args:
        low: Extremo inferior (H, S, V).
        high: Extremo superior (H, S, V).
    """
    for name, value in zip(
        ("H min", "H max", "S min", "S max", "V min", "V max"),
        (low[0], high[0], low[1], high[1], low[2], high[2]),
    ):
        cv2.setTrackbarPos(name, WINDOW_NAME, int(value))


def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    """Muestrea el color bajo el cursor al hacer clic.

    Funciona en cualquiera de las tres vistas: se toma el resto de dividir por
    el ancho de una vista, asi que da igual si se pincha en el original, en la
    mascara o en el resultado.

    Args:
        event: Tipo de evento de raton.
        x: Coordenada x en la ventana.
        y: Coordenada y en la ventana.
        flags: Banderas del evento, sin usar.
        param: Parametro de usuario, sin usar.
    """
    global _extra_range
    if event != cv2.EVENT_LBUTTONDOWN or _last_frame is None:
        return

    # De coordenadas de la ventana a coordenadas del fotograma.
    frame_x = int((x % PANE_WIDTH) / _scale)
    frame_y = int(y / _scale)
    height, width = _last_frame.shape[:2]
    half = SAMPLE_PATCH // 2
    x0, x1 = max(0, frame_x - half), min(width, frame_x + half + 1)
    y0, y1 = max(0, frame_y - half), min(height, frame_y + half + 1)

    hsv = cv2.cvtColor(_last_frame, cv2.COLOR_BGR2HSV)
    try:
        ranges = sample_hsv_range(hsv[y0:y1, x0:x1])
    except ValueError as error:
        _logger.warning("%s", error)
        return

    set_trackbars(*ranges[0])
    _extra_range = ranges[1] if len(ranges) > 1 else None
    if _extra_range is not None:
        _logger.info(
            "El tono cruza la costura del circulo: es un rojo. Se usan dos rangos."
        )
    # Se imprime YA, no solo al salir: asi la medida no se pierde pase lo que
    # pase despues, y se puede copiar sin cerrar la ventana.
    print(format_ranges(_color, ranges), flush=True)


def create_window(initial: Tuple[int, int, int, int, int, int]) -> None:
    """Crea la ventana con los seis deslizadores y el callback del raton.

    Args:
        initial: Valores iniciales en el orden (H min, H max, S min, S max,
            V min, V max).
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    for (name, maximum), value in zip(TRACKBARS, initial):
        cv2.createTrackbar(name, WINDOW_NAME, int(value), maximum, _noop)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)


def read_trackbars() -> HsvRange:
    """Lee el rango HSV actual de los deslizadores.

    Returns:
        El par ((H,S,V) minimo, (H,S,V) maximo).
    """
    values = [cv2.getTrackbarPos(name, WINDOW_NAME) for name, _ in TRACKBARS]
    return (values[0], values[2], values[4]), (values[1], values[3], values[5])


def current_ranges() -> Tuple[HsvRange, ...]:
    """Devuelve los rangos activos, que son dos si el color cruza la costura.

    Returns:
        Uno o dos rangos.
    """
    ranges: List[HsvRange] = [read_trackbars()]
    if _extra_range is not None:
        ranges.append(_extra_range)
    return tuple(ranges)


def build_mask(frame: Image, ranges: Tuple[HsvRange, ...]) -> Image:
    """Construye la mascara uniendo todos los rangos activos.

    Practica 1: cv2.inRange por rango y cv2.bitwise_or para unirlos.

    Args:
        frame: Fotograma BGR.
        ranges: Rangos activos.

    Returns:
        La mascara binaria.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        piece = cv2.inRange(
            hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)
        )
        mask = cv2.bitwise_or(mask, piece)
    return mask


def build_view(frame: Image, mask: Image) -> Image:
    """Compone el fotograma, la mascara y el resultado enmascarado.

    Args:
        frame: Fotograma BGR.
        mask: Mascara binaria.

    Returns:
        Las tres vistas una al lado de la otra, ya escaladas.
    """
    global _scale
    _scale = PANE_WIDTH / float(frame.shape[1])
    size = (PANE_WIDTH, max(int(round(frame.shape[0] * _scale)), 1))

    masked = cv2.bitwise_and(frame, frame, mask=mask)
    panes = [
        cv2.resize(frame, size, interpolation=cv2.INTER_AREA),
        cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), size, interpolation=cv2.INTER_AREA),
        cv2.resize(masked, size, interpolation=cv2.INTER_AREA),
    ]
    for pane, text in zip(panes, ("original", "mascara", "resultado")):
        cv2.putText(
            pane, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )
    return np.hstack(panes)


def print_result(color: str, ranges: Tuple[HsvRange, ...]) -> None:
    """Imprime los rangos elegidos listos para pegar en la configuracion.

    Args:
        color: Nombre del color ajustado.
        ranges: Rangos activos.
    """
    print()
    print("Pega esta linea en DEFAULT_HSV_RANGES, en src/lanzarote_tracker/config.py:")
    print()
    print(format_ranges(color, ranges))
    print()
    if len(ranges) == 1 and color == "red":
        print(
            "AVISO: para el rojo esperaba DOS rangos y solo hay uno. Vuelve a "
            "hacer clic en el centro de la figura roja, no en su borde."
        )
        print()


def run(config: AppConfig, image_path: Optional[Path], color: str) -> int:
    """Ejecuta el ajuste, sobre la webcam o sobre una imagen fija.

    Args:
        config: Configuracion de la aplicacion.
        image_path: Imagen sobre la que ajustar. Si es None se usa la webcam.
        color: Nombre del color que se esta ajustando, para el mensaje final.

    Returns:
        0 si termina correctamente, 1 si no hay fuente de imagen.
    """
    global _last_frame, _color

    _color = color
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
    _logger.info(
        "Haz CLIC sobre la figura de color para medir su rango. "
        "Sal con 'q' o ESC para imprimirlo."
    )

    # Se guarda el ultimo rango visto DENTRO del bucle. No se puede leer los
    # deslizadores despues, porque para entonces la ventana ya no existe y
    # cv2.getTrackbarPos falla justo al ir a imprimir el resultado, que es lo
    # unico que se queria de toda la ejecucion.
    ranges: Tuple[HsvRange, ...] = (((0, 0, 0), (MAX_HUE, 255, 255)),)

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

            _last_frame = frame
            ranges = current_ranges()
            cv2.imshow(WINDOW_NAME, build_view(frame, build_mask(frame, ranges)))

            if is_quit_key(cv2.waitKey(30) & 0xFF, config.camera.quit_keys):
                break
    except KeyboardInterrupt:
        _logger.info("Interrumpido por el usuario.")
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()

    print_result(color, ranges)
    return 0


def main() -> int:
    """Punto de entrada del script.

    Returns:
        El codigo de salida del proceso.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Mide el rango HSV de una tarjeta. Haz clic sobre la figura de "
            "color y el rango se calcula solo; los deslizadores quedan para "
            "retocarlo. Al salir imprime la linea lista para pegar en config.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "El rojo cruza la costura del circulo cromatico y necesita DOS "
            "rangos: al hacer clic se detecta y se imprimen los dos."
        ),
    )
    config = get_config()
    parser.add_argument(
        "--color",
        default="red",
        choices=sorted(config.security.hsv_ranges),
        help="Color que se esta ajustando, para el mensaje final.",
    )
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
    return run(config, args.image, args.color)


if __name__ == "__main__":
    raise SystemExit(main())
