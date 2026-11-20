"""Cliente de la API de OpenSky Network.

De OpenSky se usan tres cosas distintas:

1. **Vectores de estado** en un recuadro geografico alrededor del aeropuerto:
   dicen que aviones hay ahora mismo sobre GCRR y con que callsign.
2. **Base de datos de aeronaves**: a partir del identificador ICAO24 del avion
   se obtienen su modelo y su matricula, que son datos FIJOS.
3. **Llegadas y salidas por aeropuerto**: dan origen, destino y horas. Estas
   consultas solo estan disponibles AL DIA SIGUIENTE, asi que el sistema no
   puede completar la ficha con el vuelo recien grabado: hay que volver sobre
   el clip al dia siguiente.

La ficha solo muestra datos FIJOS del avion (modelo, matricula, origen,
destino). NO se muestran altura, velocidad ni posicion, aunque los vectores de
estado los traigan.

TODO IMPORTANTE: verificar en la documentacion ACTUAL de OpenSky
(https://openskynetwork.github.io/opensky-api/) que metodo de autenticacion
sigue vigente y cuales son los limites de uso vigentes, y ajustar este cliente a
lo que se confirme. Aqui no se da por supuesto ningun esquema ni ningun limite.

Las credenciales se leen SIEMPRE de variables de entorno (`os.environ`), nunca
del codigo: los nombres estan documentados en `.env.example`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from lanzarote_tracker.config import IdentificationConfig


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


class OpenSkyClient:
    """Cliente HTTP de OpenSky Network.

    Attributes:
        config: Configuracion de identificacion, con la URL base, el recuadro
            geografico y los nombres de las variables de entorno.
    """

    def __init__(self, config: IdentificationConfig) -> None:
        """Inicializa el cliente leyendo las credenciales del entorno.

        Las credenciales se leen con `os.environ.get` de los nombres que indica
        la configuracion. Si no estan definidas, el cliente trabaja de forma
        anonima: OpenSky lo permite, con menos cuota.

        Args:
            config: Configuracion de identificacion.
        """
        self.config: IdentificationConfig = config
        self._username: Optional[str] = os.environ.get(config.username_env_var)
        self._password: Optional[str] = os.environ.get(config.password_env_var)

    @property
    def has_credentials(self) -> bool:
        """True si hay usuario y contrasena en el entorno."""
        return bool(self._username and self._password)

    def get_states_in_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        at_time: Optional[datetime] = None,
    ) -> List[StateVector]:
        """Pide los vectores de estado de los aviones dentro del recuadro.

        Endpoint ``/states/all`` con los parametros ``lamin``, ``lamax``,
        ``lomin`` y ``lomax``.

        TODO: comprobar en la documentacion actual si la consulta historica por
        marca de tiempo sigue disponible y para que usuarios, antes de apoyarse
        en `at_time`.

        Args:
            bbox: Recuadro (lat_min, lat_max, lon_min, lon_max). Si es None se
                usa `config.bbox`, el del aeropuerto de Lanzarote.
            at_time: Momento para el que se piden los estados. Si es None se
                piden los actuales.

        Returns:
            Los vectores de estado de los aviones del recuadro.

        Raises:
            RuntimeError: Si la peticion falla o la respuesta no es valida.
        """
        raise NotImplementedError(
            "Pendiente: OpenSkyClient.get_states_in_bbox "
            "(identification/opensky_client.py). requests.get a /states/all. "
            "Antes: verificar autenticacion y limites en la documentacion actual."
        )

    def get_aircraft_metadata(self, icao24: str) -> Optional[AircraftMetadata]:
        """Pide los datos fijos de una aeronave: modelo y matricula.

        TODO: confirmar en la documentacion actual como se consulta la base de
        datos de aeronaves (endpoint, formato y si requiere autenticacion), y
        si conviene descargarla entera una vez y cachearla en local en vez de
        consultarla avion a avion.

        Args:
            icao24: Identificador ICAO24 en hexadecimal.

        Returns:
            Los datos de la aeronave, o None si no esta en la base de datos.

        Raises:
            RuntimeError: Si la peticion falla.
        """
        raise NotImplementedError(
            "Pendiente: OpenSkyClient.get_aircraft_metadata "
            "(identification/opensky_client.py). "
            "Antes: verificar el endpoint en la documentacion actual."
        )

    def get_arrivals(self, begin: datetime, end: datetime) -> List[FlightRecord]:
        """Pide las llegadas al aeropuerto en un intervalo.

        Endpoint ``/flights/arrival`` con ``airport``, ``begin`` y ``end`` en
        marcas de tiempo UNIX.

        AVISO: estas consultas solo estan disponibles AL DIA SIGUIENTE del
        vuelo, asi que no sirven para completar la ficha en el momento.

        Args:
            begin: Inicio del intervalo, en UTC.
            end: Fin del intervalo, en UTC.

        Returns:
            Los vuelos que aterrizaron en el aeropuerto en ese intervalo.

        Raises:
            RuntimeError: Si la peticion falla.
        """
        raise NotImplementedError(
            "Pendiente: OpenSkyClient.get_arrivals (identification/opensky_client.py). "
            "requests.get a /flights/arrival. Disponible solo al dia siguiente."
        )

    def get_departures(self, begin: datetime, end: datetime) -> List[FlightRecord]:
        """Pide las salidas del aeropuerto en un intervalo.

        Endpoint ``/flights/departure``. Mismo aviso que :meth:`get_arrivals`:
        solo esta disponible al dia siguiente.

        Args:
            begin: Inicio del intervalo, en UTC.
            end: Fin del intervalo, en UTC.

        Returns:
            Los vuelos que despegaron del aeropuerto en ese intervalo.

        Raises:
            RuntimeError: Si la peticion falla.
        """
        raise NotImplementedError(
            "Pendiente: OpenSkyClient.get_departures (identification/opensky_client.py). "
            "requests.get a /flights/departure. Disponible solo al dia siguiente."
        )


__all__ = [
    "StateVector",
    "AircraftMetadata",
    "FlightRecord",
    "OpenSkyClient",
]
