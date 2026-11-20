"""Helpers reutilizables de lectura y escritura de imagenes y video.

Estilo de las practicas de la asignatura: identificadores en ingles, docstrings
en espanol, type hints en todo y helpers del tipo ``load_images`` /
``show_image`` / ``write_image`` para no repetir codigo de OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from lanzarote_tracker.config import CameraConfig

#: Una imagen es siempre un array de NumPy en BGR (o en escala de grises).
Image = np.ndarray
PathLike = Union[str, Path]

#: Extensiones de imagen que reconoce `load_images`.
IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_image(path: PathLike, grayscale: bool = False) -> Image:
    """Carga una imagen del disco.

    Args:
        path: Ruta del fichero de imagen.
        grayscale: Si se carga en escala de grises en vez de en color BGR.

    Returns:
        La imagen como array de NumPy.

    Raises:
        FileNotFoundError: Si el fichero no existe o OpenCV no sabe leerlo.
    """
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return image


def load_images(
    paths: Union[PathLike, Iterable[PathLike]],
    grayscale: bool = False,
    pattern: str = "*",
) -> List[Image]:
    """Carga varias imagenes, desde un directorio o desde una lista de rutas.

    Args:
        paths: Directorio con imagenes, o iterable de rutas de fichero.
        grayscale: Si se cargan en escala de grises.
        pattern: Patron glob aplicado cuando `paths` es un directorio.

    Returns:
        Lista de imagenes en el orden alfabetico de sus rutas.

    Raises:
        FileNotFoundError: Si el directorio no existe o alguna imagen no se lee.
    """
    return [load_image(path, grayscale=grayscale) for path in list_image_paths(paths, pattern)]


def list_image_paths(
    paths: Union[PathLike, Iterable[PathLike]],
    pattern: str = "*",
) -> List[Path]:
    """Resuelve un directorio o un iterable de rutas en una lista ordenada de ficheros.

    Args:
        paths: Directorio con imagenes, o iterable de rutas de fichero.
        pattern: Patron glob aplicado cuando `paths` es un directorio.

    Returns:
        Lista ordenada de rutas de imagen existentes.

    Raises:
        FileNotFoundError: Si `paths` es un directorio que no existe.
    """
    if isinstance(paths, (str, Path)):
        directory = Path(paths)
        if not directory.is_dir():
            raise FileNotFoundError(f"No es un directorio: {directory}")
        return sorted(
            item
            for item in directory.glob(pattern)
            if item.suffix.lower() in IMAGE_EXTENSIONS
        )
    return [Path(item) for item in paths]


def write_image(path: PathLike, image: Image) -> Path:
    """Guarda una imagen creando los directorios intermedios que hagan falta.

    Args:
        path: Ruta de destino, incluida la extension.
        image: Imagen a guardar.

    Returns:
        La ruta donde se ha guardado la imagen.

    Raises:
        IOError: Si OpenCV no consigue escribir el fichero.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"No se pudo escribir la imagen: {path}")
    return path


def show_image(
    image: Image,
    window_name: str = "image",
    wait: int = 0,
    destroy: bool = True,
) -> int:
    """Muestra una imagen en una ventana de OpenCV.

    Args:
        image: Imagen a mostrar.
        window_name: Titulo de la ventana.
        wait: Milisegundos de espera; 0 bloquea hasta que se pulse una tecla.
        destroy: Si se cierra la ventana al terminar.

    Returns:
        El codigo de la tecla pulsada, ya enmascarado con 0xFF.
    """
    cv2.imshow(window_name, image)
    key = cv2.waitKey(wait) & 0xFF
    if destroy:
        cv2.destroyWindow(window_name)
    return key


def resolve_camera_backend(backend_name: str) -> int:
    """Traduce el nombre de un backend de captura a su constante de OpenCV.

    `cv2.CAP_DSHOW` solo existe en Windows, asi que el nombre se resuelve en
    tiempo de ejecucion y se cae a `cv2.CAP_ANY` si no esta disponible.

    Args:
        backend_name: Nombre de la constante, p. ej. "CAP_DSHOW" o "CAP_ANY".

    Returns:
        El valor numerico de la constante, o `cv2.CAP_ANY` si no existe.
    """
    return int(getattr(cv2, backend_name, cv2.CAP_ANY))


def open_camera(config: Optional[CameraConfig] = None) -> cv2.VideoCapture:
    """Abre la webcam con el backend y la resolucion de la configuracion.

    Args:
        config: Configuracion de camara. Si es None se usa la de por defecto.

    Returns:
        Un `cv2.VideoCapture` ya abierto.

    Raises:
        RuntimeError: Si la camara no se puede abrir.
    """
    config = config or CameraConfig()
    capture = cv2.VideoCapture(config.index, resolve_camera_backend(config.backend))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"No se pudo abrir la camara {config.index} con el backend "
            f"{config.backend}. Comprueba que no la esta usando otra aplicacion."
        )
    if config.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    if config.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    if config.fps:
        capture.set(cv2.CAP_PROP_FPS, config.fps)
    return capture


def open_video(path: PathLike) -> cv2.VideoCapture:
    """Abre un fichero de video.

    Args:
        path: Ruta del video.

    Returns:
        Un `cv2.VideoCapture` ya abierto.

    Raises:
        FileNotFoundError: Si el fichero no existe.
        RuntimeError: Si OpenCV no puede abrirlo.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el video: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV no pudo abrir el video: {path}")
    return capture


def video_properties(capture: cv2.VideoCapture) -> Tuple[int, int, float, int]:
    """Lee las propiedades basicas de un `cv2.VideoCapture`.

    Args:
        capture: Captura ya abierta.

    Returns:
        Una tupla (ancho, alto, fps, numero de fotogramas). El numero de
        fotogramas es 0 en un directo, donde no se conoce de antemano.
    """
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, max(frame_count, 0)


def is_quit_key(key: int, quit_keys: Sequence[int] = (ord("q"), 27)) -> bool:
    """Indica si la tecla pulsada debe cerrar la ventana.

    Args:
        key: Codigo devuelto por `cv2.waitKey`, ya enmascarado con 0xFF.
        quit_keys: Teclas que cierran. Por defecto 'q' y ESC.

    Returns:
        True si hay que salir del bucle de video.
    """
    return key in tuple(quit_keys)


__all__ = [
    "IMAGE_EXTENSIONS",
    "Image",
    "PathLike",
    "load_image",
    "load_images",
    "list_image_paths",
    "write_image",
    "show_image",
    "resolve_camera_backend",
    "open_camera",
    "open_video",
    "video_properties",
    "is_quit_key",
]
