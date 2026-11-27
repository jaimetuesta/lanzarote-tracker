"""Cliente de adsbdb: modelo, matricula, aerolinea y ruta de un vuelo.

De aqui salen los DATOS FIJOS de la ficha. Se consulta en tres pasos, porque no
todos los vuelos traen todo:

1. ``/aircraft/<icao24>`` con el indicativo: da la aeronave y, muchas veces, su
   ruta.
2. ``/callsign/<indicativo>``: si el primero no dio ruta.
3. ``/airline/<prefijo>``: si no hay aerolinea, se deduce de las tres primeras
   letras del indicativo, que son el codigo OACI de la compania.

AVISO sobre la ruta: adsbdb devuelve la ruta HABITUAL de ese indicativo, no la
del vuelo de hoy. Por eso :mod:`lanzarote_tracker.identification.flight_info`
solo la acepta si uno de sus extremos es Lanzarote y ademas cuadra con el
movimiento observado.

Fuera de las practicas: es una consulta a una base de datos externa, no una
tecnica de vision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import requests

from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import IdentificationConfig

_logger = get_logger(__name__)

#: Respuesta cruda de adsbdb para una aeronave o una ruta.
RawRecord = Dict[str, Any]


@dataclass(frozen=True)
class AircraftRoute:
    """Lo que adsbdb sabe de un vuelo.

    Attributes:
        aircraft: Datos de la aeronave (matricula, fabricante, tipo).
        route: Ruta habitual del indicativo, con origen y destino.
        airline: Nombre de la compania, si se ha podido resolver.
    """

    aircraft: RawRecord = field(default_factory=dict)
    route: RawRecord = field(default_factory=dict)
    airline: Optional[str] = None

    @property
    def registration(self) -> Optional[str]:
        """Matricula de la aeronave, p. ej. "G-UZHE"."""
        return self.aircraft.get("registration")

    @property
    def icao_type(self) -> Optional[str]:
        """Codigo OACI del tipo de aeronave, p. ej. "A20N"."""
        return self.aircraft.get("icao_type")

    @property
    def origin(self) -> RawRecord:
        """Aeropuerto de origen de la ruta habitual."""
        return self.route.get("origin") or {}

    @property
    def destination(self) -> RawRecord:
        """Aeropuerto de destino de la ruta habitual."""
        return self.route.get("destination") or {}


class AdsbdbClient:
    """Consulta adsbdb y cachea las respuestas.

    La cache importa: durante un aterrizaje el mismo avion aparece en decenas de
    consultas seguidas, y sus datos fijos no cambian.

    Attributes:
        config: Configuracion de identificacion.
    """

    def __init__(self, config: IdentificationConfig) -> None:
        """Inicializa el cliente con la cache vacia.

        Args:
            config: Configuracion de identificacion.
        """
        self.config: IdentificationConfig = config
        self._cache: Dict[Tuple[str, str], AircraftRoute] = {}

    def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Any:
        """Lanza una peticion a adsbdb y devuelve el campo `response`.

        Args:
            path: Ruta sin la barra inicial, p. ej. "aircraft/4074b5".
            params: Parametros de consulta.

        Returns:
            El contenido de `response`, o None si la peticion falla. adsbdb
            responde 404 cuando simplemente no conoce el vuelo, que es un caso
            normal y no un error.
        """
        try:
            response = requests.get(
                f"{self.config.adsbdb_url}/{path}",
                params=params,
                timeout=self.config.request_timeout_seconds,
            )
            if not response.ok:
                return None
            return response.json().get("response")
        except (requests.RequestException, ValueError) as error:
            _logger.debug("adsbdb %s ha fallado: %s", path, error)
            return None

    def lookup(self, hexcode: str, flight_callsign: str = "") -> AircraftRoute:
        """Busca los datos fijos de un avion y su ruta.

        Args:
            hexcode: Identificador ICAO24 en hexadecimal.
            flight_callsign: Indicativo del vuelo, si se conoce. Mejora mucho el
                resultado: sin el no hay ruta ni aerolinea.

        Returns:
            Lo que se haya podido resolver. Los campos que adsbdb no conozca
            quedan vacios; nunca lanza excepcion por un vuelo desconocido.
        """
        key = (hexcode, flight_callsign)
        if key in self._cache:
            return self._cache[key]

        aircraft: RawRecord = {}
        route: RawRecord = {}

        payload = self._get(
            f"aircraft/{hexcode}",
            params={"callsign": flight_callsign} if flight_callsign else None,
        )
        if isinstance(payload, dict):
            aircraft = payload.get("aircraft") or {}
            route = payload.get("flightroute") or {}

        if not route and flight_callsign:
            payload = self._get(f"callsign/{flight_callsign}")
            if isinstance(payload, dict):
                route = payload.get("flightroute") or {}

        airline = (route.get("airline") or {}).get("name")
        if not airline and flight_callsign:
            # Las tres primeras letras del indicativo son el codigo OACI de la
            # compania: EZY5701 -> EZY -> easyJet.
            payload = self._get(f"airline/{flight_callsign[:3]}")
            if isinstance(payload, list) and payload:
                airline = payload[0].get("name")

        result = AircraftRoute(aircraft=aircraft, route=route, airline=airline)
        self._cache[key] = result
        return result

    def clear_cache(self) -> None:
        """Vacia la cache de consultas."""
        self._cache.clear()


__all__ = ["RawRecord", "AircraftRoute", "AdsbdbClient"]
