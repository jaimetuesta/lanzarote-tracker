"""Deduccion de llegada o salida a partir de los datos ADS-B.

CUIDADO CON LA ARQUITECTURA. Este modulo deduce si un avion llega o sale **sin
mirar la imagen**: lo saca del historial de alturas que emite el propio avion.
Eso NO sustituye a
:mod:`lanzarote_tracker.classification.arrival_departure`, que es donde el
sistema lo deduce POR VISION (frente/espaldas combinado con el sentido de uso
de la pista) y que es lo que pide el enunciado. Son dos fuentes independientes
del mismo hecho:

- la de vision es la que se evalua y la que va al apartado de resultados;
- esta es un dato externo.

Y precisamente por ser independiente, esta sirve para dos cosas muy utiles:

1. **Etiquetar el dataset automaticamente.** Cruzando la marca de tiempo UTC de
   cada clip con el movimiento deducido aqui se obtiene la etiqueta
   llegada/salida sin anotarla a mano, y de ahi la de frente/espaldas.
2. **Medir el clasificador de vision** contra una referencia que no procede de
   las imagenes.

NUNCA se debe usar esta deduccion para rellenar la salida del clasificador de
vision: seria evaluar el sistema contra si mismo.

Fuera de las practicas: la deduccion por historial de alturas no es una tecnica
de vision. En el informe se describe como fuente de verdad de referencia para
el entrenamiento y la evaluacion.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from lanzarote_tracker.classification.arrival_departure import FlightPhase
from lanzarote_tracker.config import IdentificationConfig
from lanzarote_tracker.identification.adsb_client import (
    RawAircraft,
    altitude_ft,
    vertical_rate_fpm,
)

#: Una muestra del historial: (marca de tiempo UNIX, altura en pies).
AltitudeSample = Tuple[int, int]
#: Extremos de una pista, como ((lat, lon), (lat, lon)).
RunwayEndpoints = Tuple[Tuple[float, float], Tuple[float, float]]

#: Metros por grado de latitud. Basta esta aproximacion plana para distancias
#: de unos cientos de metros alrededor de la pista.
METERS_PER_DEGREE = 111320.0


def distance_to_runway(lat: float, lon: float, runway: RunwayEndpoints) -> float:
    """Distancia en metros de un punto al eje de la pista.

    Se proyecta el punto sobre el SEGMENTO que une los dos umbrales, no sobre
    la recta infinita que los contiene: asi un avion alineado con la pista pero
    a varios kilometros por delante no cuenta como que esta en ella.

    Args:
        lat: Latitud del avion, en grados.
        lon: Longitud del avion, en grados.
        runway: Los dos extremos de la pista.

    Returns:
        La distancia en metros al punto mas cercano del segmento.
    """
    (lat1, lon1), (lat2, lon2) = runway
    cos_lat = math.cos(math.radians(lat1))
    point_x = (lon - lon1) * cos_lat * METERS_PER_DEGREE
    point_y = (lat - lat1) * METERS_PER_DEGREE
    axis_x = (lon2 - lon1) * cos_lat * METERS_PER_DEGREE
    axis_y = (lat2 - lat1) * METERS_PER_DEGREE

    squared_length = axis_x * axis_x + axis_y * axis_y
    if squared_length == 0:
        return math.hypot(point_x, point_y)

    # Proyeccion escalar recortada a [0, 1] para quedarse dentro del segmento.
    projection = (point_x * axis_x + point_y * axis_y) / squared_length
    projection = max(0.0, min(1.0, projection))
    return math.hypot(point_x - projection * axis_x, point_y - projection * axis_y)


def is_on_runway(
    aircraft: RawAircraft,
    config: Optional[IdentificationConfig] = None,
) -> bool:
    """Indica si el avion esta operando en la pista del aeropuerto.

    Cuenta como "en pista" un avion en el aire a baja altura, y uno en tierra
    que este sobre la pista, aunque este parado en cabecera esperando para
    despegar.

    Si no hay coordenadas de pista configuradas
    (`config.runway_endpoints` es None), no se sabe donde esta la pista y solo
    cuentan los aviones en tierra que van rapido, que es bastante mas impreciso:
    deja fuera a los que esperan parados en cabecera.

    Args:
        aircraft: Avion tal cual lo devuelve la API ADS-B.
        config: Configuracion de identificacion.

    Returns:
        True si el avion esta operando en la pista.
    """
    config = config or IdentificationConfig()
    altitude = altitude_ft(aircraft, config)
    if altitude is None or altitude >= config.max_altitude_ft:
        return False
    if altitude > 0:
        return True

    if config.runway_endpoints is not None and aircraft.get("lat") is not None:
        distance = distance_to_runway(
            float(aircraft["lat"]), float(aircraft["lon"]), config.runway_endpoints
        )
        return distance < config.runway_margin_m

    ground_speed = aircraft.get("gs") or 0
    return float(ground_speed) >= config.min_ground_speed_kt


def infer_movement(
    history: Sequence[AltitudeSample],
) -> Tuple[FlightPhase, Optional[int]]:
    """Deduce llegada o salida buscando el ultimo cambio entre aire y tierra.

    Se recorre el historial desde el final: el cambio mas reciente es el que
    describe lo que el avion acaba de hacer. Un avion que aterriza pasa de
    altura positiva a cero; uno que despega, de cero a altura positiva.

    Args:
        history: Historial de alturas del avion, en orden cronologico.

    Returns:
        Una tupla (fase, marca de tiempo del evento). La fase es UNKNOWN y la
        marca None si en el historial no hay ningun cambio entre aire y tierra.
    """
    for (time_before, alt_before), (time_after, alt_after) in zip(
        reversed(history[:-1]), reversed(history[1:])
    ):
        if alt_before > 0 and alt_after == 0:
            return FlightPhase.ARRIVAL, time_after
        if alt_before == 0 and alt_after > 0:
            return FlightPhase.DEPARTURE, time_before
    return FlightPhase.UNKNOWN, None


def infer_from_vertical_rate(
    aircraft: RawAircraft,
    config: Optional[IdentificationConfig] = None,
) -> FlightPhase:
    """Deduce la fase de un avion en el aire por si sube o baja.

    Es el recurso para cuando no se ha visto el despegue ni el aterrizaje,
    porque el sistema empezo a mirar a mitad de la maniobra.

    Args:
        aircraft: Avion tal cual lo devuelve la API ADS-B.
        config: Configuracion de identificacion.

    Returns:
        DEPARTURE si sube, ARRIVAL si baja, UNKNOWN si esta en tierra o si la
        velocidad vertical no llega a `config.min_vertical_rate_fpm`, umbral que
        evita confundir el ruido de la medida con una maniobra.
    """
    config = config or IdentificationConfig()
    altitude = altitude_ft(aircraft, config)
    if altitude is None or altitude <= 0:
        return FlightPhase.UNKNOWN
    rate = vertical_rate_fpm(aircraft)
    if abs(rate) < config.min_vertical_rate_fpm:
        return FlightPhase.UNKNOWN
    return FlightPhase.DEPARTURE if rate > 0 else FlightPhase.ARRIVAL


class AltitudeHistory:
    """Historial de alturas por avion, con reinicio por inactividad.

    Si un avion desaparece y vuelve horas despues, su historial antiguo
    describe otra operacion distinta: mezclarlos haria deducir un movimiento
    que no ha ocurrido. Por eso se vacia tras
    `config.history_reset_seconds` sin noticias.

    Attributes:
        config: Configuracion de identificacion.
    """

    def __init__(self, config: IdentificationConfig) -> None:
        """Inicializa el historial vacio.

        Args:
            config: Configuracion de identificacion.
        """
        self.config: IdentificationConfig = config
        self._samples: dict = {}

    def record(self, hexcode: str, timestamp: int, altitude: int) -> List[AltitudeSample]:
        """Anade una muestra al historial de un avion.

        Args:
            hexcode: Identificador ICAO24 del avion.
            timestamp: Marca de tiempo UNIX de la muestra.
            altitude: Altura en pies, con 0 para "en tierra".

        Returns:
            El historial completo del avion tras anadir la muestra.
        """
        samples = self._samples.setdefault(hexcode, [])
        if samples and timestamp - samples[-1][0] > self.config.history_reset_seconds:
            samples.clear()
        samples.append((timestamp, altitude))
        return samples

    def get(self, hexcode: str) -> List[AltitudeSample]:
        """Devuelve el historial de un avion.

        Args:
            hexcode: Identificador ICAO24 del avion.

        Returns:
            El historial, o una lista vacia si no hay ninguno.
        """
        return self._samples.get(hexcode, [])

    def append_landing(self, hexcode: str, timestamp: int) -> List[AltitudeSample]:
        """Anota un aterrizaje supuesto al final del historial de un avion.

        Se usa cuando se deja de ver a un avion que iba bajando a muy baja
        altura: lo mas probable es que haya aterrizado y por eso ha dejado de
        emitir.

        Args:
            hexcode: Identificador ICAO24 del avion.
            timestamp: Marca de tiempo UNIX que se le asigna al aterrizaje.

        Returns:
            El historial completo del avion.
        """
        samples = self._samples.setdefault(hexcode, [])
        samples.append((timestamp, 0))
        return samples

    def clear(self, hexcode: Optional[str] = None) -> None:
        """Olvida el historial de un avion, o el de todos.

        Args:
            hexcode: Avion a olvidar. Si es None se olvidan todos.
        """
        if hexcode is None:
            self._samples.clear()
        else:
            self._samples.pop(hexcode, None)


__all__ = [
    "AltitudeSample",
    "RunwayEndpoints",
    "METERS_PER_DEGREE",
    "distance_to_runway",
    "is_on_runway",
    "infer_movement",
    "infer_from_vertical_rate",
    "AltitudeHistory",
]
