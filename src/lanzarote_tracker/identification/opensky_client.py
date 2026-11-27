"""Cliente de la API de OpenSky Network. FUENTE SECUNDARIA.

La fuente primaria de la identificacion es ADS-B
(:mod:`lanzarote_tracker.identification.adsb_client`), porque responde al
momento. OpenSky se conserva por dos motivos:

- sus consultas de llegadas y salidas por aeropuerto dan origen, destino y
  horas CONFIRMADOS, no la ruta habitual del indicativo que devuelve adsbdb;
- su cobertura es distinta, asi que sirve para contrastar.

Pero esas consultas **solo estan disponibles al dia siguiente** del vuelo, asi
que no sirven para rellenar la ficha mientras se graba: hay que volver sobre el
clip al dia siguiente, cruzando la marca de tiempo UTC de sus metadatos.

AUTENTICACION: OpenSky usa **OAuth2 `client_credentials`**, no usuario y
contrasena. Se piden un `client_id` y un `client_secret` en el panel de la
cuenta, se cambian por un token en el endpoint de autenticacion y el token se
manda en la cabecera `Authorization: Bearer`. Las credenciales se leen de
variables de entorno; `credentials.json` NO se usa y no debe subirse al
repositorio.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from lanzarote_tracker.common.log import get_logger
from lanzarote_tracker.config import IdentificationConfig

_logger = get_logger(__name__)


@dataclass(frozen=True)
class StateVector:
    """Vector de estado de un avion devuelto por OpenSky.

    Solo se conservan los campos que el proyecto necesita. La altura, la
    velocidad y la posicion NO se muestran en la ficha, pero la posicion si se
    usa internamente para emparejar el avion de la imagen con el vuelo.

    Attributes:
        icao24: Identificador ICAO24 del transpondedor, en hexadecimal.
        callsign: Indicativo del vuelo, ya sin espacios sobrantes.
        latitude: Latitud en grados, o None si OpenSky no la da.
        longitude: Longitud en grados, o None si OpenSky no la da.
        on_ground: True si el avion esta en tierra.
        time_position: Marca de tiempo UNIX de la posicion.
    """

    icao24: str
    callsign: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    on_ground: bool = False
    time_position: Optional[int] = None

    @classmethod
    def from_row(cls, row: List[Any]) -> "StateVector":
        """Construye un vector de estado desde la fila cruda de OpenSky.

        El endpoint `/states/all` devuelve listas posicionales, no objetos: los
        indices son los que documenta OpenSky (0 icao24, 1 callsign,
        3 time_position, 5 longitude, 6 latitude, 8 on_ground).

        Args:
            row: Fila tal cual la devuelve OpenSky.

        Returns:
            El vector de estado con los campos que usa el proyecto.
        """

        def at(index: int) -> Any:
            return row[index] if len(row) > index else None

        return cls(
            icao24=str(at(0) or ""),
            callsign=str(at(1) or "").strip(),
            latitude=at(6),
            longitude=at(5),
            on_ground=bool(at(8)),
            time_position=at(3),
        )


@dataclass(frozen=True)
class AircraftMetadata:
    """Datos fijos de una aeronave, de la base de datos de OpenSky.

    Attributes:
        icao24: Identificador ICAO24.
        registration: Matricula, p. ej. "EC-MXY".
        model: Modelo del avion, p. ej. "Airbus A320-214".
        manufacturer: Fabricante.
        operator: Compania operadora.
    """

    icao24: str
    registration: str = ""
    model: str = ""
    manufacturer: str = ""
    operator: str = ""


@dataclass(frozen=True)
class FlightRecord:
    """Un vuelo de llegada o salida del aeropuerto.

    Attributes:
        icao24: Identificador ICAO24 del avion.
        callsign: Indicativo del vuelo.
        departure_airport: Codigo ICAO del aeropuerto de origen.
        arrival_airport: Codigo ICAO del aeropuerto de destino.
        first_seen: Marca de tiempo UNIX del despegue.
        last_seen: Marca de tiempo UNIX del aterrizaje.
    """

    icao24: str
    callsign: str = ""
    departure_airport: str = ""
    arrival_airport: str = ""
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "FlightRecord":
        """Construye un registro de vuelo desde la respuesta de OpenSky.

        Args:
            payload: Objeto de la lista que devuelven `/flights/arrival` y
                `/flights/departure`.

        Returns:
            El registro con los campos que usa el proyecto.
        """
        return cls(
            icao24=str(payload.get("icao24") or ""),
            callsign=str(payload.get("callsign") or "").strip(),
            departure_airport=str(payload.get("estDepartureAirport") or ""),
            arrival_airport=str(payload.get("estArrivalAirport") or ""),
            first_seen=payload.get("firstSeen"),
            last_seen=payload.get("lastSeen"),
        )


class OpenSkyClient:
    """Cliente HTTP de OpenSky Network con autenticacion OAuth2.

    Attributes:
        config: Configuracion de identificacion.
    """

    def __init__(self, config: IdentificationConfig) -> None:
        """Inicializa el cliente leyendo las credenciales del entorno.

        Las credenciales se leen con `os.environ.get` de los nombres que indica
        la configuracion. Si no estan definidas, el cliente trabaja de forma
        anonima: OpenSky lo permite, con menos cuota y sin acceso a las
        consultas de llegadas y salidas.

        Args:
            config: Configuracion de identificacion.
        """
        self.config: IdentificationConfig = config
        self._client_id: Optional[str] = os.environ.get(config.client_id_env_var)
        self._client_secret: Optional[str] = os.environ.get(config.client_secret_env_var)
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def has_credentials(self) -> bool:
        """True si hay `client_id` y `client_secret` en el entorno."""
        return bool(self._client_id and self._client_secret)

    def _access_token(self) -> Optional[str]:
        """Devuelve un token valido, pidiendo uno nuevo si hace falta.

        El token se guarda hasta poco antes de caducar
        (`config.token_refresh_margin_seconds`), para no pedir uno en cada
        consulta.

        Returns:
            El token, o None si no hay credenciales o si no se puede obtener.
        """
        if not self.has_credentials:
            return None
        if self._token and time.time() < self._token_expires_at:
            return self._token

        try:
            response = requests.post(
                self.config.opensky_token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            _logger.warning("No se ha podido obtener el token de OpenSky: %s", error)
            return None

        self._token = payload.get("access_token")
        lifetime = float(payload.get("expires_in", 0))
        self._token_expires_at = (
            time.time() + lifetime - self.config.token_refresh_margin_seconds
        )
        return self._token

    def _headers(self) -> Dict[str, str]:
        """Construye la cabecera de autorizacion, si hay token.

        Returns:
            La cabecera `Authorization`, o un diccionario vacio para una
            consulta anonima.
        """
        token = self._access_token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _get(self, path: str, params: Dict[str, Any]) -> Optional[Any]:
        """Lanza una consulta GET a OpenSky.

        Args:
            path: Ruta sin la barra inicial, p. ej. "states/all".
            params: Parametros de consulta.

        Returns:
            La respuesta ya decodificada, o None si la consulta falla.
        """
        try:
            response = requests.get(
                f"{self.config.opensky_base_url}/{path}",
                params=params,
                headers=self._headers(),
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            _logger.warning("La consulta a OpenSky %s ha fallado: %s", path, error)
            return None

    def get_states_in_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[StateVector]:
        """Pide los vectores de estado de los aviones dentro del recuadro.

        Args:
            bbox: Recuadro (lat_min, lat_max, lon_min, lon_max). Si es None se
                usa `config.bbox`, el del aeropuerto de Lanzarote.

        Returns:
            Los vectores de estado de los aviones del recuadro. Lista vacia si
            la consulta falla o no hay aviones.
        """
        lat_min, lat_max, lon_min, lon_max = bbox or self.config.bbox
        payload = self._get(
            "states/all",
            {"lamin": lat_min, "lamax": lat_max, "lomin": lon_min, "lomax": lon_max},
        )
        if not isinstance(payload, dict):
            return []
        return [StateVector.from_row(row) for row in payload.get("states") or []]

    def get_arrivals(self, begin: datetime, end: datetime) -> List[FlightRecord]:
        """Pide las llegadas al aeropuerto en un intervalo.

        AVISO: solo esta disponible AL DIA SIGUIENTE del vuelo.

        Args:
            begin: Inicio del intervalo, en UTC.
            end: Fin del intervalo, en UTC.

        Returns:
            Los vuelos que aterrizaron en el aeropuerto en ese intervalo.
        """
        return self._flights("arrival", begin, end)

    def get_departures(self, begin: datetime, end: datetime) -> List[FlightRecord]:
        """Pide las salidas del aeropuerto en un intervalo.

        AVISO: solo esta disponible AL DIA SIGUIENTE del vuelo.

        Args:
            begin: Inicio del intervalo, en UTC.
            end: Fin del intervalo, en UTC.

        Returns:
            Los vuelos que despegaron del aeropuerto en ese intervalo.
        """
        return self._flights("departure", begin, end)

    def _flights(
        self, kind: str, begin: datetime, end: datetime
    ) -> List[FlightRecord]:
        """Consulta `/flights/arrival` o `/flights/departure`.

        Args:
            kind: "arrival" o "departure".
            begin: Inicio del intervalo, en UTC.
            end: Fin del intervalo, en UTC.

        Returns:
            Los vuelos del intervalo. Lista vacia si la consulta falla, cosa que
            tambien ocurre si el intervalo es de hoy: estos datos solo estan
            disponibles al dia siguiente.
        """
        payload = self._get(
            f"flights/{kind}",
            {
                "airport": self.config.airport_icao,
                "begin": int(begin.replace(tzinfo=begin.tzinfo or timezone.utc).timestamp()),
                "end": int(end.replace(tzinfo=end.tzinfo or timezone.utc).timestamp()),
            },
        )
        if not isinstance(payload, list):
            return []
        return [FlightRecord.from_payload(item) for item in payload]

    def get_aircraft_metadata(self, icao24: str) -> Optional[AircraftMetadata]:
        """Pide los datos fijos de una aeronave: modelo y matricula.

        TODO: la base de datos de aeronaves de OpenSky se consulta aparte de la
        API REST y conviene descargarla entera una vez y cachearla en local en
        vez de consultarla avion a avion. Mientras tanto, estos datos se
        obtienen de adsbdb, que es la fuente primaria del proyecto
        (:mod:`lanzarote_tracker.identification.adsbdb_client`).

        Args:
            icao24: Identificador ICAO24 en hexadecimal.

        Returns:
            Los datos de la aeronave, o None si no se pueden obtener.
        """
        raise NotImplementedError(
            "No implementado a proposito: los datos fijos de la aeronave se "
            "obtienen de adsbdb (AdsbdbClient.lookup), que es la fuente "
            "primaria del proyecto y no necesita credenciales."
        )


__all__ = [
    "StateVector",
    "AircraftMetadata",
    "FlightRecord",
    "OpenSkyClient",
]
