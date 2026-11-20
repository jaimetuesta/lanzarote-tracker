"""Comprueba que todos los modulos del paquete se importan sin error.

Es un test barato que detecta errores de sintaxis, imports circulares y
nombres mal escritos en los stubs. Importar un modulo NO ejecuta sus
`raise NotImplementedError`, que solo saltan al llamar a las funciones.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import List

import pytest

import lanzarote_tracker


def iter_module_names() -> List[str]:
    """Lista todos los modulos del paquete.

    Returns:
        Los nombres completos de los modulos, en orden alfabetico.
    """
    return sorted(
        module.name
        for module in pkgutil.walk_packages(
            lanzarote_tracker.__path__, prefix="lanzarote_tracker."
        )
    )


@pytest.mark.parametrize("module_name", iter_module_names())
def test_module_imports(module_name: str) -> None:
    """Cada modulo del paquete se importa sin lanzar excepciones.

    Args:
        module_name: Nombre completo del modulo.
    """
    importlib.import_module(module_name)


def test_package_exposes_version() -> None:
    """El paquete declara su version."""
    assert isinstance(lanzarote_tracker.__version__, str)
    assert lanzarote_tracker.__version__


def test_factories_reject_unknown_backends() -> None:
    """Las fabricas rechazan un backend desconocido con ValueError."""
    from dataclasses import replace

    from lanzarote_tracker.classification.factory import create_classifier
    from lanzarote_tracker.config import get_config
    from lanzarote_tracker.detection.factory import create_detector
    from lanzarote_tracker.tracking.factory import create_tracker

    config = get_config()
    with pytest.raises(ValueError):
        create_detector(replace(config.detection, backend="inventado"))
    with pytest.raises(ValueError):
        create_tracker(replace(config.tracking, backend="inventado"))
    with pytest.raises(ValueError):
        create_classifier(replace(config.classification, backend="inventado"))


def test_deep_learning_backends_are_disabled() -> None:
    """Los backends de deep learning fallan con un mensaje claro.

    Siguen desactivados mientras su uso este pendiente de aprobacion del
    profesor.
    """
    from dataclasses import replace

    from lanzarote_tracker.classification.factory import create_classifier
    from lanzarote_tracker.config import get_config
    from lanzarote_tracker.detection.factory import create_detector

    config = get_config()
    with pytest.raises(NotImplementedError, match="aprobacion del profesor"):
        create_detector(replace(config.detection, backend="coco"))
    with pytest.raises(NotImplementedError, match="aprobacion del profesor"):
        create_classifier(replace(config.classification, backend="cnn"))


def test_extensions_are_not_implemented() -> None:
    """Las ampliaciones fallan indicando que no son alcance base."""
    from lanzarote_tracker.extensions.kalman import KalmanBoxFilter
    from lanzarote_tracker.extensions.live import LivePipeline

    with pytest.raises(NotImplementedError, match="AMPLIACION"):
        KalmanBoxFilter()
    with pytest.raises(NotImplementedError, match="AMPLIACION"):
        LivePipeline()
