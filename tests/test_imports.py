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


def test_deep_learning_backends_can_be_built() -> None:
    """Los backends de deep learning ya se pueden construir.

    El profesor aprobo su uso, asi que la fabrica deja de rechazarlos. Construir
    el objeto NO importa ningun framework de deep learning: los imports van
    dentro de los metodos, para que el entorno base siga sin necesitarlos.
    """
    from dataclasses import replace

    from lanzarote_tracker.classification.factory import create_classifier
    from lanzarote_tracker.config import get_config
    from lanzarote_tracker.detection.factory import create_detector

    config = get_config()
    assert create_detector(replace(config.detection, backend="coco")).name == "coco"
    assert (
        create_classifier(replace(config.classification, backend="cnn")).name == "cnn"
    )


def test_deep_learning_backends_are_not_implemented_yet() -> None:
    """Usarlos todavia falla, indicando que la linea principal es la clasica."""
    from dataclasses import replace

    import numpy as np

    from lanzarote_tracker.classification.factory import create_classifier
    from lanzarote_tracker.config import get_config
    from lanzarote_tracker.detection.factory import create_detector

    config = get_config()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    detector = create_detector(replace(config.detection, backend="coco"))
    with pytest.raises(NotImplementedError, match="classic"):
        detector.detect(frame)

    classifier = create_classifier(replace(config.classification, backend="cnn"))
    with pytest.raises(NotImplementedError, match="bow"):
        classifier.predict(frame)


def test_classical_backends_remain_the_default() -> None:
    """Aprobar el deep learning NO cambia los backends por defecto.

    El proyecto sigue siendo de vision por ordenador tradicional: `coco` y
    `cnn` son alternativas opcionales con las que comparar en el informe.
    """
    from lanzarote_tracker.config import get_config

    config = get_config()
    assert config.detection.backend == "classic"
    assert config.classification.backend == "bow"


def test_extensions_are_not_implemented() -> None:
    """Las ampliaciones fallan indicando que no son alcance base."""
    from lanzarote_tracker.extensions.kalman import KalmanBoxFilter
    from lanzarote_tracker.extensions.live import LivePipeline

    with pytest.raises(NotImplementedError, match="AMPLIACION"):
        KalmanBoxFilter()
    with pytest.raises(NotImplementedError, match="AMPLIACION"):
        LivePipeline()
