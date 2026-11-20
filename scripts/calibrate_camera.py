"""Calibra la webcam a partir de las fotos del tablero de `calibration/images/`.

Guarda K, los coeficientes de distorsion y el RMS en `calibration/results/`,
en .npz (para volver a cargarlos) y en .json (para pegarlos en el informe), y
escribe en `results/calibration/` la figura antes/despues que pide el informe.

Uso::

    python scripts/calibrate_camera.py
    python scripts/calibrate_camera.py --pattern 9x6
    python scripts/calibrate_camera.py --detect-pattern
    python scripts/calibrate_camera.py --debug

Si no sabes cuantas esquinas internas tiene tu tablero, ejecuta primero
``--detect-pattern``: prueba los tamanos habituales sobre la primera foto y te
dice cual encaja.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lanzarote_tracker.calibration.calibrate import (
    aggregate_rms,
    reprojection_errors,
    run_calibration,
)
from lanzarote_tracker.calibration.chessboard import (
    build_object_points,
    collect_corners,
    detect_pattern_size,
    parse_pattern_size,
)
from lanzarote_tracker.calibration.store import save_calibration
from lanzarote_tracker.calibration.undistort import Undistorter, side_by_side
from lanzarote_tracker.common.debug import DebugRecorder
from lanzarote_tracker.common.io_utils import list_image_paths, load_image, write_image
from lanzarote_tracker.common.log import get_logger, setup_logging
from lanzarote_tracker.config import AppConfig, get_config

_logger = get_logger(__name__)


def run_detect_pattern(images_dir: Path, max_size: int) -> int:
    """Prueba tamanos de patron sobre la primera foto y los imprime.

    Args:
        images_dir: Directorio con las fotos del tablero.
        max_size: Numero maximo de esquinas internas por lado que se prueba.

    Returns:
        0 si encuentra algun tamano, 1 si no.
    """
    paths = list_image_paths(images_dir)
    if not paths:
        _logger.error(
            "No hay fotos en %s. Tomalas con "
            "`python scripts/capture_calibration_images.py`.",
            images_dir,
        )
        return 1

    _logger.info("Probando tamanos de patron sobre %s ...", paths[0].name)
    matches = detect_pattern_size(load_image(paths[0]), max_size=max_size)
    if not matches:
        _logger.error(
            "No se ha reconocido ningun tablero en %s. Comprueba que se vea "
            "entero, bien iluminado y sin reflejos, y prueba con otra foto.",
            paths[0].name,
        )
        return 1

    best = matches[0]
    print()
    print(f"Tamanos que encajan en {paths[0].name}: {matches}")
    print(f"El mas probable es el mayor: {best[0]}x{best[1]} esquinas internas.")
    if len(matches) > 1:
        print(
            "Han encajado varios: quedate con el mayor y comprueba que coincide "
            "con tu tablero contando los cuadrados y restando uno por lado."
        )
    print()
    print("Calibra con:")
    print(f"    python scripts/calibrate_camera.py --pattern {best[0]}x{best[1]}")
    print()
    print("Y si quieres fijarlo, pon en config.py:")
    print(f"    pattern_size: Tuple[int, int] = ({best[0]}, {best[1]})")
    print()
    return 0


def run_calibrate(config: AppConfig, images_dir: Path, alpha: float) -> int:
    """Calibra, guarda el resultado y escribe la figura antes/despues.

    Args:
        config: Configuracion de la aplicacion.
        images_dir: Directorio con las fotos del tablero.
        alpha: Parametro de recorte de cv2.getOptimalNewCameraMatrix.

    Returns:
        0 si la calibracion termina, 1 si falla.
    """
    recorder = DebugRecorder("calibration", config.debug)

    try:
        result = run_calibration(images_dir, config.calibration, recorder=recorder)
        npz_path, json_path = save_calibration(result, config.paths, config.calibration)
    except (FileNotFoundError, ValueError) as error:
        _logger.error("%s", error)
        return 1

    print()
    print(result.summary())
    print()

    # Error por imagen: sirve para localizar fotos malas y para la tabla del informe.
    try:
        images = [load_image(path) for path in list_image_paths(images_dir)]
        object_points, image_points, _ = collect_corners(images, config.calibration)
        errors = reprojection_errors(result, object_points, image_points)
        worst = max(range(len(errors)), key=lambda index: errors[index])
        best = min(range(len(errors)), key=lambda index: errors[index])
        print(f"RMS por imagen: mejor {errors[best]:.4f} px (la {best}), "
              f"peor {errors[worst]:.4f} px (la {worst})")
        print(f"RMS global recompuesto: {aggregate_rms(errors):.4f} px")
        print(
            "Si una foto destaca muy por encima del resto, borrala de "
            "calibration/images/ y vuelve a calibrar."
        )
        print()
    except (ValueError, FileNotFoundError) as error:
        _logger.warning("No se ha podido calcular el error por imagen: %s", error)

    # Figura antes/despues para el informe, siempre, no solo con --debug.
    try:
        first = load_image(list_image_paths(images_dir)[0])
        undistorter = Undistorter(result, alpha=alpha)
        comparison = side_by_side(first, undistorter(first))
        figure_path = write_image(
            Path(config.paths.results) / "calibration" / "undistort_comparison.png",
            comparison,
        )
        recorder.record("original", first)
        recorder.record("undistorted", undistorter(first))
        _logger.info("Figura antes/despues en %s", figure_path)
    except (ValueError, IndexError) as error:
        _logger.warning("No se ha podido componer la figura antes/despues: %s", error)

    recorder.save_all()
    _logger.info("Calibracion guardada en %s y %s", npz_path, json_path)
    return 0


def main() -> int:
    """Punto de entrada del script.

    Returns:
        0 si la calibracion termina, 1 si falla.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calibra la webcam con las fotos del tablero y guarda K, los "
            "coeficientes de distorsion y el RMS en calibration/results/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "El tamano del patron son ESQUINAS INTERNAS (cuadrados menos uno "
            "por lado): un tablero de 10x7 cuadrados es 9x6. Si no lo sabes, "
            "usa --detect-pattern."
        ),
    )
    config = get_config()
    parser.add_argument(
        "--images",
        type=Path,
        default=config.paths.calibration_images,
        help="Directorio con las fotos del tablero.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        metavar="COLSxROWS",
        help=(
            "Esquinas internas del tablero, p. ej. '9x6'. Por defecto se usa el "
            "valor de config.py."
        ),
    )
    parser.add_argument(
        "--square",
        type=float,
        default=config.calibration.square_size_mm,
        help=(
            "Lado del cuadrado en milimetros. Solo fija la escala: no afecta a "
            "K ni a los coeficientes de distorsion."
        ),
    )
    parser.add_argument(
        "--detect-pattern",
        action="store_true",
        help="Probar tamanos de patron sobre la primera foto y salir, sin calibrar.",
    )
    parser.add_argument(
        "--max-pattern",
        type=int,
        default=12,
        help="Esquinas internas maximas por lado que prueba --detect-pattern.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=config.calibration.min_images,
        help="Fotos validas minimas para dar la calibracion por buena.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Recorte de getOptimalNewCameraMatrix: 0 recorta los bordes, 1 los conserva.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Guarda la secuencia de transformacion en results/calibration/.",
    )
    args = parser.parse_args()

    setup_logging(config.debug)

    if args.detect_pattern:
        return run_detect_pattern(args.images, args.max_pattern)

    try:
        pattern_size = (
            parse_pattern_size(args.pattern)
            if args.pattern
            else config.calibration.pattern_size
        )
    except ValueError as error:
        _logger.error("%s", error)
        return 1

    config = replace(
        config,
        calibration=replace(
            config.calibration,
            pattern_size=pattern_size,
            square_size_mm=args.square,
            min_images=args.min_images,
        ),
        debug=replace(config.debug, enabled=args.debug),
    )
    _logger.info(
        "Patron: %dx%d esquinas internas, cuadrado de %.1f mm",
        pattern_size[0],
        pattern_size[1],
        args.square,
    )
    return run_calibrate(config, args.images, args.alpha)


if __name__ == "__main__":
    raise SystemExit(main())
