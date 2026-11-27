"""Base de aeropuertos: busqueda del mas cercano a unas coordenadas.

Se usa para resolver el destino de una salida: cuando un avion que ha despegado
de Lanzarote se pierde o toca tierra, se mira que aeropuerto tiene mas cerca.

Los datos salen de OurAirports (`airports.csv`), recortados a los 9053
aeropuertos CON codigo IATA y a las columnas que hacen falta, de 13 MB a menos
de 600 KB. El fichero SI se versiona: es un dato de referencia estatico, no
material generado.

Fuera de las practicas: no es una tecnica de vision, es una busqueda por
proximidad sobre una tabla. En el informe entra como parte del bloque de
identificacion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import IdentificationConfig, PathsConfig

_logger = get_logger(__name__)

#: Nombre del fichero dentro de `PathsConfig.reference_data`.
AIRPORTS_FILENAME = "airports.csv"
#: Columnas que se cargan del CSV.
AIRPORT_COLUMNS = ("ident", "iata_code", "name", "municipality", "latitude_deg", "longitude_deg")


def load_airports(paths: Optional[PathsConfig] = None) -> pd.DataFrame:
    """Carga la tabla de aeropuertos con codigo IATA.

    Args:
        paths: Rutas del proyecto. Si es None se usan las de por defecto.

    Returns:
        Un DataFrame con las columnas de `AIRPORT_COLUMNS`, sin filas que no
        tengan codigo IATA ni coordenadas.

    Raises:
        FileNotFoundError: Si falta el fichero de aeropuertos.
    """
    paths = paths or PathsConfig()
    path = Path(paths.reference_data) / AIRPORTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe la base de aeropuertos en {path}. Deberia venir con el "
            "repositorio; recuperala con `git checkout assets/data/airports.csv`."
        )
    frame = pd.read_csv(
        path,
        dtype={"ident": str, "iata_code": str, "name": str, "municipality": str},
        usecols=list(AIRPORT_COLUMNS),
    )
    frame = frame.dropna(subset=["iata_code", "latitude_deg", "longitude_deg"])
    return frame.reset_index(drop=True)


def nearest_airport(lat: float, lon: float, airports: pd.DataFrame) -> pd.Series:
    """Devuelve el aeropuerto mas cercano a unas coordenadas.

    La distancia se aproxima en el plano corrigiendo la longitud por el coseno
    de la latitud, que a estas escalas basta y es mucho mas rapido que una
    distancia sobre la esfera: solo hace falta saber CUAL es el mas cercano, no
    a cuantos kilometros esta.

    Args:
        lat: Latitud en grados.
        lon: Longitud en grados.
        airports: Tabla devuelta por :func:`load_airports`.

    Returns:
        La fila del aeropuerto mas cercano.

    Raises:
        ValueError: Si la tabla esta vacia.
    """
    if airports.empty:
        raise ValueError("La tabla de aeropuertos esta vacia.")
    delta_lat = airports["latitude_deg"] - lat
    delta_lon = (airports["longitude_deg"] - lon) * math.cos(math.radians(lat))
    return airports.loc[(delta_lat**2 + delta_lon**2).idxmin()]


def format_airport(
    iata: Optional[str],
    municipality: Optional[str] = None,
) -> Optional[str]:
    """Formatea un aeropuerto como "IATA (Ciudad)" para la ficha.

    Args:
        iata: Codigo IATA, p. ej. "AGP".
        municipality: Ciudad del aeropuerto, si se conoce.

    Returns:
        El texto para la ficha, o None si no hay codigo IATA.
    """
    if not iata:
        return None
    if isinstance(municipality, str) and municipality.strip():
        return f"{iata} ({municipality.strip()})"
    return str(iata)


def describe_nearest(
    lat: float,
    lon: float,
    airports: pd.DataFrame,
    config: Optional[IdentificationConfig] = None,
) -> Optional[str]:
    """Busca el aeropuerto mas cercano y lo devuelve ya formateado.

    Args:
        lat: Latitud en grados.
        lon: Longitud en grados.
        airports: Tabla de aeropuertos.
        config: Configuracion de identificacion. Si el aeropuerto mas cercano
            resulta ser el propio Lanzarote, se devuelve None: significa que el
            avion no ha llegado a ningun sitio nuevo.

    Returns:
        El aeropuerto formateado, o None si es el de partida o no se encuentra.
    """
    config = config or IdentificationConfig()
    airport = nearest_airport(lat, lon, airports)
    if airport["iata_code"] == config.airport_iata:
        return None
    return format_airport(airport["iata_code"], airport.get("municipality"))


__all__ = [
    "AIRPORTS_FILENAME",
    "AIRPORT_COLUMNS",
    "load_airports",
    "nearest_airport",
    "format_airport",
    "describe_nearest",
]
