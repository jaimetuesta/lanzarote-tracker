"""Logging comun del proyecto.

Se llama `log.py` y no `logging.py` para no confundirlo con el modulo estandar
`logging`, del que depende.
"""

from __future__ import annotations

import logging
from typing import Optional

from lanzarote_tracker.config import DebugConfig

#: Nombre raiz de todos los loggers del proyecto.
ROOT_LOGGER_NAME = "lanzarote_tracker"

_LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(config: Optional[DebugConfig] = None) -> logging.Logger:
    """Configura el logger raiz del proyecto.

    Es idempotente: si ya hay un manejador instalado no anade otro, para no
    duplicar cada linea cuando varios scripts la llaman.

    Args:
        config: Configuracion de depuracion, de la que se toma el nivel.

    Returns:
        El logger raiz del proyecto, ya configurado.
    """
    config = config or DebugConfig()
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Devuelve el logger de un modulo, colgado del logger raiz del proyecto.

    Args:
        name: Normalmente ``__name__`` del modulo que llama.

    Returns:
        Un logger hijo de `ROOT_LOGGER_NAME`.
    """
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


__all__ = ["ROOT_LOGGER_NAME", "setup_logging", "get_logger"]
