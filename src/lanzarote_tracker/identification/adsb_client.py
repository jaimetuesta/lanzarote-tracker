"""Cliente ADS-B: que aviones hay ahora mismo sobre el aeropuerto.

Es la FUENTE PRIMARIA de la identificacion. Consulta `adsb.lol` y, si falla,
`adsb.fi`: las dos sirven la misma API y son abiertas, sin credenciales. A
diferencia de las consultas de llegadas y salidas de OpenSky, que solo estan
disponibles al dia siguiente, estas responden al momento, que es lo que hace
falta mientras se graba un clip.

Fuera de las practicas: ADS-B no es una tecnica de vision por ordenador. En el
informe entra como fuente de datos del bloque de identificacion, igual que
OpenSky.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests

from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import IdentificationConfig

_logger = get_logger(__name__)

#: Un avion tal cual lo devuelve la API, sin procesar.
RawAircraft = Dict[str, Any]
#: Altura en pies, o None si el avion no la informa de forma utilizable.
Altitude = Optional[int]


@dataclass(frozen=True)
class AdsbClient:
    """Consulta los aviones de la zona en las fuentes ADS-B configuradas.

    Attributes:
        config: Configuracion de identificacion.
    """

    config: IdentificationConfig

    def query(self, route: str) -> List[RawAircraft]:
        """Lanza una consulta contra las fuentes, en orden, hasta que una responda.

        Args:
            route: Ruta de la API sin la barra inicial, p. ej.
                "lat/28.945/lon/-13.605/dist/3" o "hex/4074b5".

        Returns:
            La lista de aviones. Vacia si ninguna fuente responde o si no hay
            aviones en la zona.
        """
        for base in self.config.adsb_sources:
            try:
                response = requests.get(
                    f"{base}/{route}", timeout=self.config.request_timeout_seconds
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as error:
                _logger.debug("La fuente %s ha fallado: %s", base, error)
                continue
            # adsb.lol responde con "ac" y adsb.fi con "aircraft".
            aircraft = payload.get("ac") or payload.get("aircraft") or []
            if aircraft:
                return list(aircraft)
        return []

    def aircraft_near_airport(self) -> List[RawAircraft]:
        """Devuelve los aviones dentro del radio configurado del aeropuerto.

        Returns:
            Los aviones de la zona, sin filtrar por altura.
        """
        return self.query(
            f"lat/{self.config.airport_lat}"
            f"/lon/{self.config.airport_lon}"
            f"/dist/{self.config.search_radius_nm}"
        )

    def aircraft_by_hex(self, hexcode: str) -> Optional[RawAircraft]:
        """Busca un avion concreto por su identificador ICAO24.

        Args:
            hexcode: Identificador ICAO24 en hexadecimal.

        Returns:
            El avion, o None si ninguna fuente lo encuentra.
        """
        found = self.query(f"hex/{hexcode}")
        return found[0] if found else None


def altitude_ft(
    aircraft: RawAircraft,
    config: Optional[IdentificationConfig] = None,
) -> Altitude:
    """Devuelve la altura del avion en pies, con 0 para "en tierra".

    ADS-B informa la altura de dos maneras incompatibles: la cadena "ground", o
    un numero. Ademas hay aviones que, estando en tierra, siguen emitiendo una
    altura de unas pocas decenas de pies en vez de "ground", asi que todo lo que
    quede por debajo de `config.ground_altitude_ft` tambien se considera suelo.

    Args:
        aircraft: Avion tal cual lo devuelve la API.
        config: Configuracion de identificacion.

    Returns:
        La altura en pies, 0 si esta en tierra, o None si el campo no es
        utilizable.
    """
    config = config or IdentificationConfig()
    value: Union[str, int, float, None] = aircraft.get("alt_baro")
    if value == "ground":
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= config.ground_altitude_ft:
        return 0
    return int(value)


def vertical_rate_fpm(aircraft: RawAircraft) -> int:
    """Devuelve la velocidad vertical del avion en pies por minuto.

    Args:
        aircraft: Avion tal cual lo devuelve la API.

    Returns:
        La velocidad vertical, positiva si sube y negativa si baja. 0 si el
        avion no la informa.
    """
    rate = aircraft.get("baro_rate")
    if rate is None:
        rate = aircraft.get("geom_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return 0
    return int(rate)


def icao24(aircraft: RawAircraft) -> str:
    """Devuelve el identificador ICAO24 limpio.

    Algunas fuentes marcan con `~` los identificadores que no son ICAO reales
    (TIS-B, MLAT); el prefijo se quita para poder cruzarlos con adsbdb.

    Args:
        aircraft: Avion tal cual lo devuelve la API.

    Returns:
        El identificador en hexadecimal, sin el marcador.
    """
    return str(aircraft.get("hex", "")).strip("~")


def callsign(aircraft: RawAircraft) -> str:
    """Devuelve el indicativo del vuelo, sin espacios de relleno.

    Args:
        aircraft: Avion tal cual lo devuelve la API.

    Returns:
        El indicativo, o cadena vacia si el avion no lo emite.
    """
    return str(aircraft.get("flight") or "").strip()


__all__ = [
    "RawAircraft",
    "Altitude",
    "AdsbClient",
    "altitude_ft",
    "vertical_rate_fpm",
    "icao24",
    "callsign",
]
