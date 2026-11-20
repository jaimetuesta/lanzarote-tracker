"""Comprueba el entorno: versiones y trackers de OpenCV disponibles.

Uso::

    python scripts/check_environment.py
    python scripts/check_environment.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Dict, List, Tuple

#: Trackers que el proyecto podria querer usar, con el modulo donde buscarlos.
#: cv2.TrackerMIL_create esta siempre en el paquete base; KCF y CSRT dependen de
#: la version y de si esta instalado opencv-contrib-python (que el proyecto NO
#: usa). cv2.legacy agrupa los trackers antiguos cuando existe.
TRACKER_FACTORIES: Tuple[str, ...] = (
    "TrackerMIL_create",
    "TrackerKCF_create",
    "TrackerCSRT_create",
    "TrackerGOTURN_create",
)

LEGACY_TRACKER_FACTORIES: Tuple[str, ...] = (
    "TrackerBoosting_create",
    "TrackerMOSSE_create",
    "TrackerTLD_create",
    "TrackerMedianFlow_create",
)


def collect_versions() -> Dict[str, str]:
    """Recoge las versiones de Python y de las bibliotecas principales.

    Returns:
        Un diccionario de nombre de biblioteca a version. Las que no esten
        instaladas aparecen con el valor "NO INSTALADO".
    """
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
    }
    for module_name, key in (
        ("cv2", "opencv"),
        ("numpy", "numpy"),
        ("skimage", "scikit-image"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("imageio", "imageio"),
        ("pandas", "pandas"),
        ("yt_dlp", "yt-dlp"),
        ("requests", "requests"),
        ("pytest", "pytest"),
    ):
        try:
            module = __import__(module_name)
            versions[key] = str(getattr(module, "__version__", "desconocida"))
        except ImportError:
            versions[key] = "NO INSTALADO"
    return versions


def collect_trackers() -> Dict[str, bool]:
    """Comprueba que fabricas de trackers existen en esta instalacion de OpenCV.

    Returns:
        Un diccionario del nombre de la fabrica a si esta disponible. Las del
        modulo `cv2.legacy` se prefijan con "legacy.".
    """
    available: Dict[str, bool] = {}
    try:
        import cv2
    except ImportError:
        return {name: False for name in TRACKER_FACTORIES}

    for name in TRACKER_FACTORIES:
        available[name] = hasattr(cv2, name)

    has_legacy = hasattr(cv2, "legacy")
    available["legacy"] = has_legacy
    if has_legacy:
        for name in LEGACY_TRACKER_FACTORIES:
            available[f"legacy.{name}"] = hasattr(cv2.legacy, name)
    return available


def collect_cameras(max_index: int = 3) -> List[int]:
    """Busca indices de camara que se puedan abrir.

    Abre y cierra cada indice, asi que puede encender brevemente el piloto de
    la webcam. No se ejecuta salvo que se pase `--cameras`.

    Args:
        max_index: Ultimo indice que se prueba, sin incluirlo.

    Returns:
        Lista de indices de camara disponibles.
    """
    try:
        import cv2
    except ImportError:
        return []

    from lanzarote_tracker.common.io_utils import resolve_camera_backend
    from lanzarote_tracker.config import CameraConfig

    backend = resolve_camera_backend(CameraConfig().backend)
    found: List[int] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            found.append(index)
        capture.release()
    return found


def build_report(check_cameras: bool) -> Dict[str, object]:
    """Monta el informe completo del entorno.

    Args:
        check_cameras: Si se buscan camaras disponibles.

    Returns:
        Un diccionario con las versiones, los trackers y, si procede, las camaras.
    """
    report: Dict[str, object] = {
        "versions": collect_versions(),
        "trackers": collect_trackers(),
    }
    if check_cameras:
        report["cameras"] = collect_cameras()
    return report


def print_report(report: Dict[str, object]) -> None:
    """Imprime el informe en formato legible.

    Args:
        report: Informe devuelto por :func:`build_report`.
    """
    versions = report["versions"]
    trackers = report["trackers"]

    print("=" * 62)
    print("LanzaroteTracker - comprobacion del entorno")
    print("=" * 62)
    print("\nVersiones")
    print("-" * 62)
    for name, value in versions.items():  # type: ignore[union-attr]
        print(f"  {name:<16} {value}")

    print("\nTrackers de OpenCV disponibles")
    print("-" * 62)
    for name, available in trackers.items():  # type: ignore[union-attr]
        mark = "si" if available else "NO"
        print(f"  {name:<28} {mark}")

    if "cameras" in report:
        print("\nCamaras detectadas")
        print("-" * 62)
        cameras = report["cameras"]
        print(f"  {cameras if cameras else 'ninguna'}")

    print()


def main() -> int:
    """Punto de entrada del script.

    Returns:
        0 si OpenCV y NumPy estan disponibles, 1 si falta alguno.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Comprueba el entorno del proyecto: imprime las versiones de Python, "
            "OpenCV y NumPy y que trackers de OpenCV estan disponibles en esta "
            "instalacion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime el informe en JSON en vez de en texto.",
    )
    parser.add_argument(
        "--cameras",
        action="store_true",
        help="Busca ademas que indices de camara se pueden abrir (enciende la webcam).",
    )
    args = parser.parse_args()

    report = build_report(check_cameras=args.cameras)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)

    versions = report["versions"]
    missing = [
        name
        for name in ("opencv", "numpy")
        if versions[name] == "NO INSTALADO"  # type: ignore[index]
    ]
    if missing:
        print(f"FALTAN dependencias imprescindibles: {', '.join(missing)}")
        print("Crea el entorno con `conda env create -f environment.yml`.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
