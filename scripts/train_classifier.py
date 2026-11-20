"""Entrena el clasificador de orientacion frente/espaldas.

El backend por defecto es `bow` (bolsa de palabras SIFT + K-means + SVM,
Practica 5 apartado A), que es la linea base. El backend `cnn` esta
DESACTIVADO a la espera de la aprobacion del profesor.

El modelo entrenado se guarda en `models/`, que NO se versiona.

ESTADO: el entrenamiento todavia no esta implementado. El script arranca y
falla con un mensaje que dice que falta.

Uso::

    python scripts/train_classifier.py
    python scripts/train_classifier.py --backend bow --vocabulary 300
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lanzarote_tracker.classification.factory import create_classifier
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import get_config

_logger = get_logger(__name__)


def main() -> int:
    """Punto de entrada del script.

    Returns:
        0 si el entrenamiento termina, 1 si falla o falta por implementar.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el clasificador de orientacion del avion sobre la particion "
            "de entrenamiento y guarda el modelo en models/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "El backend 'cnn' esta desactivado: su uso de deep learning esta "
            "pendiente de aprobacion del profesor."
        ),
    )
    config = get_config()
    parser.add_argument(
        "--backend",
        choices=list(config.classification.available_backends),
        default=config.classification.backend,
        help="Backend de clasificacion.",
    )
    parser.add_argument(
        "--vocabulary",
        type=int,
        default=config.classification.vocabulary_size,
        help="Numero de palabras visuales del vocabulario (solo backend bow).",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=config.paths.models,
        help="Directorio donde se guarda el modelo entrenado.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)
    config = replace(
        config,
        classification=replace(
            config.classification,
            backend=args.backend,
            vocabulary_size=args.vocabulary,
        ),
    )

    try:
        # La fabrica valida el backend y rechaza los que estan desactivados a
        # la espera de la aprobacion del profesor, con su propio mensaje.
        classifier = create_classifier(config.classification)
    except ValueError as error:
        _logger.error("%s", error)
        return 1
    except NotImplementedError as error:
        if config.classification.backend == "bow":
            _logger.error(
                "El entrenamiento todavia no esta implementado. Faltan: "
                "capture/frames.py (dataset y splits) y classification/bow.py "
                "(SIFT + K-means + SVM)."
            )
        else:
            _logger.error("%s", error)
        return 1

    _logger.error(
        "El entrenamiento todavia no esta implementado: falta cargar el dataset "
        "de capture/frames.py y llamar a %s.fit.",
        type(classifier).__name__,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
