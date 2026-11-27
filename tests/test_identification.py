"""Tests de la identificacion del vuelo.

Ninguno toca la red: las fuentes ADS-B se sustituyen por dobles, de modo que la
bateria sigue siendo rapida y no depende de que haya aviones volando.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

from lanzarote_tracker.classification.arrival_departure import FlightPhase
from lanzarote_tracker.config import IdentificationConfig, get_config
from lanzarote_tracker.identification.adsb_client import (
    RawAircraft,
    altitude_ft,
    callsign,
    icao24,
    vertical_rate_fpm,
)
from lanzarote_tracker.identification.adsbdb_client import AircraftRoute
from lanzarote_tracker.identification.airports import (
    describe_nearest,
    format_airport,
    load_airports,
    nearest_airport,
)
from lanzarote_tracker.identification.flight_info import (
    FlightInfo,
    build_flight_info,
    infer_phase_from_route,
    resolve_model,
    resolve_route,
)
from lanzarote_tracker.identification.live_tracker import (
    RECORDS_FILENAME,
    FlightTracker,
)
from lanzarote_tracker.identification.matcher import (
    candidates_for,
    load_records,
    match_flight,
)
from lanzarote_tracker.identification.movement import (
    AltitudeHistory,
    distance_to_runway,
    infer_from_vertical_rate,
    infer_movement,
    is_on_runway,
)


@pytest.fixture()
def config() -> IdentificationConfig:
    """Configuracion de identificacion por defecto."""
    return get_config().identification


# --- adsb_client ------------------------------------------------------------
def test_altitude_ground_string_is_zero(config: IdentificationConfig) -> None:
    """ADS-B informa "ground" como cadena; se traduce a 0."""
    assert altitude_ft({"alt_baro": "ground"}, config) == 0


def test_low_altitude_counts_as_ground(config: IdentificationConfig) -> None:
    """Hay aviones en tierra que emiten unas decenas de pies en vez de "ground"."""
    assert altitude_ft({"alt_baro": config.ground_altitude_ft - 1}, config) == 0


def test_altitude_above_the_ground_threshold_is_kept(
    config: IdentificationConfig,
) -> None:
    """Por encima del umbral se conserva el valor."""
    assert altitude_ft({"alt_baro": 2500}, config) == 2500


def test_missing_altitude_is_none(config: IdentificationConfig) -> None:
    """Sin altura utilizable se devuelve None, no 0: no es lo mismo que estar en tierra."""
    assert altitude_ft({}, config) is None
    assert altitude_ft({"alt_baro": None}, config) is None


def test_vertical_rate_falls_back_to_geometric() -> None:
    """Si no hay `baro_rate` se usa `geom_rate`."""
    assert vertical_rate_fpm({"geom_rate": -800}) == -800
    assert vertical_rate_fpm({"baro_rate": 1200, "geom_rate": -800}) == 1200
    assert vertical_rate_fpm({}) == 0


def test_icao24_strips_the_non_icao_marker() -> None:
    """Las fuentes marcan con `~` los identificadores que no son ICAO reales."""
    assert icao24({"hex": "~abc123"}) == "abc123"


def test_callsign_is_stripped() -> None:
    """El indicativo viene con espacios de relleno."""
    assert callsign({"flight": "EZY5701 "}) == "EZY5701"
    assert callsign({}) == ""


# --- movement ---------------------------------------------------------------
#: Pista de prueba a lo largo del meridiano, en el ecuador para simplificar.
TEST_RUNWAY = ((0.0, 0.0), (0.0, 0.01))


def test_distance_to_runway_is_perpendicular() -> None:
    """Un punto al lado del eje esta a su distancia perpendicular."""
    distance = distance_to_runway(0.001, 0.005, TEST_RUNWAY)
    assert distance == pytest.approx(111.32, rel=1e-3)


def test_distance_is_measured_to_the_segment_not_the_line() -> None:
    """Un punto alineado pero pasado el umbral mide hasta el extremo.

    Si se midiera a la recta infinita, un avion a kilometros por delante de la
    pista contaria como que esta sobre ella.
    """
    distance = distance_to_runway(0.0, 0.02, TEST_RUNWAY)
    assert distance == pytest.approx(1113.2, rel=1e-3)


def test_airborne_at_low_altitude_is_on_runway(config: IdentificationConfig) -> None:
    """Un avion bajo y cerca cuenta como operando en la pista."""
    assert is_on_runway({"alt_baro": 800}, config) is True


def test_high_altitude_is_a_passing_flight(config: IdentificationConfig) -> None:
    """Por encima del maximo es un vuelo de paso, no una operacion."""
    assert is_on_runway({"alt_baro": config.max_altitude_ft + 1}, config) is False


def test_slow_aircraft_on_ground_is_not_on_runway(
    config: IdentificationConfig,
) -> None:
    """Sin coordenadas de pista, un avion lento en tierra se da por aparcado."""
    assert config.runway_endpoints is None
    assert is_on_runway({"alt_baro": "ground", "gs": 10}, config) is False


def test_fast_aircraft_on_ground_is_on_runway(config: IdentificationConfig) -> None:
    """Sin coordenadas de pista, la velocidad es el unico criterio disponible."""
    aircraft = {"alt_baro": "ground", "gs": config.min_ground_speed_kt + 5}
    assert is_on_runway(aircraft, config) is True


def test_runway_coordinates_catch_an_aircraft_waiting_to_depart(
    config: IdentificationConfig,
) -> None:
    """Con coordenadas de pista se reconoce al avion parado en cabecera.

    Es el caso que el criterio de velocidad deja fuera.
    """
    with_runway = replace(config, runway_endpoints=TEST_RUNWAY)
    waiting = {"alt_baro": "ground", "gs": 2, "lat": 0.0, "lon": 0.0005}
    assert is_on_runway(waiting, config) is False
    assert is_on_runway(waiting, with_runway) is True


def test_infer_movement_detects_a_landing() -> None:
    """Pasar de altura positiva a cero es un aterrizaje."""
    phase, moment = infer_movement([(10, 1500), (20, 600), (30, 0)])
    assert phase is FlightPhase.ARRIVAL
    assert moment == 30


def test_infer_movement_detects_a_takeoff() -> None:
    """Pasar de cero a altura positiva es un despegue."""
    phase, moment = infer_movement([(10, 0), (20, 0), (30, 900)])
    assert phase is FlightPhase.DEPARTURE
    assert moment == 20


def test_infer_movement_uses_the_latest_transition() -> None:
    """Con un aterrizaje y luego un despegue, vale el mas reciente."""
    history = [(10, 1200), (20, 0), (30, 0), (40, 800)]
    phase, _ = infer_movement(history)
    assert phase is FlightPhase.DEPARTURE


def test_infer_movement_without_transitions_is_unknown() -> None:
    """Un avion que solo vuela no dice si llega o sale."""
    phase, moment = infer_movement([(10, 3000), (20, 3000)])
    assert phase is FlightPhase.UNKNOWN
    assert moment is None


def test_vertical_rate_distinguishes_climb_from_descent(
    config: IdentificationConfig,
) -> None:
    """Subiendo es una salida y bajando una llegada."""
    climbing = {"alt_baro": 2000, "baro_rate": 1500}
    descending = {"alt_baro": 2000, "baro_rate": -1500}
    assert infer_from_vertical_rate(climbing, config) is FlightPhase.DEPARTURE
    assert infer_from_vertical_rate(descending, config) is FlightPhase.ARRIVAL


def test_small_vertical_rate_is_not_a_maneuver(config: IdentificationConfig) -> None:
    """Por debajo del umbral es ruido de la medida, no una maniobra."""
    level = {"alt_baro": 2000, "baro_rate": config.min_vertical_rate_fpm - 1}
    assert infer_from_vertical_rate(level, config) is FlightPhase.UNKNOWN


def test_history_resets_after_a_long_gap(config: IdentificationConfig) -> None:
    """Un avion que vuelve horas despues empieza con el historial limpio.

    Si no, se mezclarian dos operaciones distintas y se deduciria un movimiento
    que no ha ocurrido.
    """
    history = AltitudeHistory(config)
    history.record("abc", 0, 1000)
    samples = history.record("abc", config.history_reset_seconds + 1, 0)
    assert samples == [(config.history_reset_seconds + 1, 0)]


def test_history_keeps_samples_within_the_window(
    config: IdentificationConfig,
) -> None:
    """Dentro de la ventana el historial se acumula."""
    history = AltitudeHistory(config)
    history.record("abc", 0, 1000)
    samples = history.record("abc", 30, 0)
    assert samples == [(0, 1000), (30, 0)]


# --- airports ---------------------------------------------------------------
@pytest.fixture()
def airports() -> pd.DataFrame:
    """Tabla de aeropuertos reducida para los tests."""
    return pd.DataFrame(
        [
            {
                "ident": "GCRR",
                "iata_code": "ACE",
                "name": "Cesar Manrique-Lanzarote",
                "municipality": "San Bartolome",
                "latitude_deg": 28.945499,
                "longitude_deg": -13.6052,
            },
            {
                "ident": "LEMG",
                "iata_code": "AGP",
                "name": "Malaga",
                "municipality": "Malaga",
                "latitude_deg": 36.6749,
                "longitude_deg": -4.4991,
            },
        ]
    )


def test_real_airport_table_contains_lanzarote() -> None:
    """La base de aeropuertos versionada trae GCRR."""
    frame = load_airports()
    assert not frame.empty
    assert (frame["ident"] == "GCRR").any()


def test_nearest_airport_finds_the_closest(airports: pd.DataFrame) -> None:
    """Se elige el aeropuerto mas cercano a las coordenadas."""
    found = nearest_airport(36.7, -4.5, airports)
    assert found["iata_code"] == "AGP"


def test_format_airport_adds_the_city() -> None:
    """La ficha muestra "IATA (Ciudad)" cuando se conoce la ciudad."""
    assert format_airport("AGP", "Malaga") == "AGP (Malaga)"
    assert format_airport("AGP", None) == "AGP"
    assert format_airport(None) is None


def test_describe_nearest_ignores_the_home_airport(
    airports: pd.DataFrame, config: IdentificationConfig
) -> None:
    """Volver a tierra en Lanzarote no es llegar a un destino nuevo."""
    assert describe_nearest(28.945, -13.605, airports, config) is None


# --- flight_info ------------------------------------------------------------
def make_route(origin: str = "", destination: str = "", **kwargs) -> AircraftRoute:
    """Construye una ruta de adsbdb para los tests.

    Args:
        origin: Codigo IATA del origen.
        destination: Codigo IATA del destino.
        **kwargs: Campos extra de :class:`AircraftRoute`.

    Returns:
        La ruta.
    """
    route: Dict = {}
    if origin:
        route["origin"] = {"iata_code": origin, "municipality": "Ciudad"}
    if destination:
        route["destination"] = {"iata_code": destination, "municipality": "Ciudad"}
    return AircraftRoute(route=route, **kwargs)


def test_resolve_model_prefers_the_readable_name() -> None:
    """Un tipo conocido se muestra con su nombre comercial."""
    route = AircraftRoute(aircraft={"icao_type": "A20N"})
    assert resolve_model(route) == "Airbus A320neo"


def test_resolve_model_falls_back_to_manufacturer_and_type() -> None:
    """Un tipo desconocido se compone con lo que da adsbdb."""
    route = AircraftRoute(
        aircraft={"icao_type": "ZZZZ", "manufacturer": "Boeing", "type": "787-9"}
    )
    assert resolve_model(route) == "Boeing 787-9"


def test_resolve_model_falls_back_to_the_icao_code() -> None:
    """Sin nada mejor, el codigo OACI al menos identifica el aparato."""
    assert resolve_model(AircraftRoute(aircraft={"icao_type": "ZZZZ"})) == "ZZZZ"


def test_route_is_accepted_when_it_matches_the_arrival(
    config: IdentificationConfig,
) -> None:
    """Una llegada con destino ACE cuadra: se usa su origen."""
    route = make_route(origin="SEN", destination="ACE")
    origin, destination = resolve_route(route, FlightPhase.ARRIVAL, config)
    assert origin == "SEN (Ciudad)"
    assert destination == "ACE (Ciudad)"


def test_route_is_discarded_when_it_contradicts_the_observation(
    config: IdentificationConfig,
) -> None:
    """Una llegada cuya ruta habitual SALE de ACE no es la de hoy.

    adsbdb devuelve la ruta habitual del indicativo. Creersela llenaria la
    ficha de datos falsos, asi que se descarta entera y solo se conserva el
    extremo que se sabe con certeza: que el avion esta en Lanzarote.
    """
    route = make_route(origin="ACE", destination="AGP")
    origin, destination = resolve_route(route, FlightPhase.ARRIVAL, config)
    assert origin == ""
    assert destination == config.airport_label


def test_route_not_touching_lanzarote_is_discarded(
    config: IdentificationConfig,
) -> None:
    """Una ruta que no pasa por ACE no puede ser la del vuelo observado."""
    route = make_route(origin="MAD", destination="BCN")
    origin, destination = resolve_route(route, FlightPhase.DEPARTURE, config)
    assert origin == config.airport_label
    assert destination == ""


def test_phase_from_route_reads_the_lanzarote_end(
    config: IdentificationConfig,
) -> None:
    """Si la ruta habitual pasa por ACE, su extremo dice si llega o sale."""
    assert (
        infer_phase_from_route(make_route("SEN", "ACE"), config) is FlightPhase.ARRIVAL
    )
    assert (
        infer_phase_from_route(make_route("ACE", "AGP"), config)
        is FlightPhase.DEPARTURE
    )
    assert (
        infer_phase_from_route(make_route("MAD", "BCN"), config) is FlightPhase.UNKNOWN
    )


def test_card_never_shows_altitude_speed_or_position(
    config: IdentificationConfig,
) -> None:
    """El enunciado solo admite datos FIJOS en la ficha.

    Las fuentes ADS-B traen altura, velocidad y posicion, asi que conviene
    fijar por test que no se cuelan en la ficha.
    """
    info = build_flight_info(
        hexcode="4074b5",
        flight_callsign="EZY5701",
        route=make_route("SEN", "ACE", aircraft={"icao_type": "A20N"}),
        phase=FlightPhase.ARRIVAL,
        config=config,
        event_time=1795195343,
        aircraft={"alt_baro": 1200, "gs": 180, "lat": 28.9, "lon": -13.6},
    )
    rendered = " ".join(f"{label} {value}" for label, value in info.as_lines(config))
    for forbidden in ("1200", "180", "28.9", "-13.6"):
        assert forbidden not in rendered


def test_card_lines_follow_the_configured_fields(
    config: IdentificationConfig,
) -> None:
    """La ficha muestra exactamente los campos de `card_fields`, en su orden."""
    info = FlightInfo(callsign="EZY5701", phase=FlightPhase.ARRIVAL)
    lines = info.as_lines(config)
    assert len(lines) == len(config.card_fields)
    assert lines[0][0] == config.card_labels["movement"]


def test_event_time_label_depends_on_the_phase(
    config: IdentificationConfig,
) -> None:
    """La misma hora se etiqueta como despegue o aterrizaje segun la fase."""
    arrival = FlightInfo(phase=FlightPhase.ARRIVAL, event_time=1795195343)
    departure = FlightInfo(phase=FlightPhase.DEPARTURE, event_time=1795195343)
    assert arrival.event_time_label(config) == "Hora de aterrizaje"
    assert departure.event_time_label(config) == "Hora de despegue"


def test_unknown_fields_show_the_placeholder(config: IdentificationConfig) -> None:
    """Un campo vacio se muestra con el marcador, no en blanco."""
    lines = dict(FlightInfo().as_lines(config))
    assert lines[config.card_labels["registration"]] == config.unknown_value


# --- matcher ----------------------------------------------------------------
def write_records(directory: Path, rows: List[Dict]) -> Path:
    """Escribe un fichero de fichas para los tests.

    Args:
        directory: Directorio donde escribirlo.
        rows: Fichas a escribir.

    Returns:
        La ruta del fichero.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RECORDS_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_records_round_trip(tmp_path: Path, config: IdentificationConfig) -> None:
    """Las fichas guardadas se vuelven a leer con su fase."""
    paths = replace(get_config().paths, flight_records=tmp_path)
    write_records(
        tmp_path,
        [{"t": 1000, "icao24": "abc", "movement": "llegada", "callsign": "EZY1"}],
    )
    records = load_records(paths)
    assert len(records) == 1
    assert records[0].info.phase is FlightPhase.ARRIVAL
    assert records[0].info.callsign == "EZY1"


def test_missing_records_file_is_not_an_error(tmp_path: Path) -> None:
    """Sin fichas registradas se devuelve una lista vacia, no una excepcion."""
    paths = replace(get_config().paths, flight_records=tmp_path / "vacio")
    assert load_records(paths) == []


def test_match_picks_the_closest_in_time(
    tmp_path: Path, config: IdentificationConfig
) -> None:
    """Entre varias fichas compatibles gana la mas cercana al fotograma."""
    paths = replace(get_config().paths, flight_records=tmp_path)
    write_records(
        tmp_path,
        [
            {"t": 1000, "icao24": "lejos", "movement": "llegada"},
            {"t": 1180, "icao24": "cerca", "movement": "llegada"},
        ],
    )
    frame_time = datetime.fromtimestamp(1200, tz=timezone.utc)
    best = match_flight(frame_time, FlightPhase.ARRIVAL, config, paths=paths)
    assert best is not None
    assert best.info.icao24 == "cerca"


def test_match_prefers_the_phase_that_vision_deduced(
    tmp_path: Path, config: IdentificationConfig
) -> None:
    """Una ficha algo mas lejana pero de la fase correcta gana a una cercana."""
    paths = replace(get_config().paths, flight_records=tmp_path)
    write_records(
        tmp_path,
        [
            {"t": 1195, "icao24": "otra_fase", "movement": "salida"},
            {"t": 1150, "icao24": "misma_fase", "movement": "llegada"},
        ],
    )
    frame_time = datetime.fromtimestamp(1200, tz=timezone.utc)
    best = match_flight(frame_time, FlightPhase.ARRIVAL, config, paths=paths)
    assert best is not None
    assert best.info.icao24 == "misma_fase"


def test_match_outside_the_window_finds_nothing(
    tmp_path: Path, config: IdentificationConfig
) -> None:
    """Una ficha de hace horas no corresponde al clip."""
    paths = replace(get_config().paths, flight_records=tmp_path)
    write_records(tmp_path, [{"t": 1000, "icao24": "vieja", "movement": "llegada"}])
    frame_time = datetime.fromtimestamp(99000, tz=timezone.utc)
    assert match_flight(frame_time, FlightPhase.ARRIVAL, config, paths=paths) is None


def test_candidates_are_ordered_by_phase_then_distance(
    config: IdentificationConfig,
) -> None:
    """El orden es: primero los de la fase correcta, luego los mas cercanos."""
    from lanzarote_tracker.identification.matcher import RecordedCard

    records = [
        RecordedCard(1190, FlightInfo(icao24="a", phase=FlightPhase.DEPARTURE)),
        RecordedCard(1100, FlightInfo(icao24="b", phase=FlightPhase.ARRIVAL)),
        RecordedCard(1150, FlightInfo(icao24="c", phase=FlightPhase.ARRIVAL)),
    ]
    frame_time = datetime.fromtimestamp(1200, tz=timezone.utc)
    found = candidates_for(records, frame_time, FlightPhase.ARRIVAL, config)
    assert [candidate.info.icao24 for candidate in found] == ["c", "b", "a"]


# --- FlightTracker ----------------------------------------------------------
class FakeAdsb:
    """Doble del cliente ADS-B que sirve respuestas fijas."""

    def __init__(self, aircraft: List[RawAircraft]) -> None:
        """Inicializa el doble.

        Args:
            aircraft: Aviones que devolvera la consulta de la zona.
        """
        self.aircraft = aircraft

    def aircraft_near_airport(self) -> List[RawAircraft]:
        """Devuelve los aviones configurados."""
        return self.aircraft

    def aircraft_by_hex(self, hexcode: str) -> Optional[RawAircraft]:
        """Devuelve None: en estos tests no se sigue a ninguna salida."""
        return None


class FakeAdsbdb:
    """Doble de adsbdb que devuelve siempre la misma aeronave."""

    def __init__(self, route: Optional[AircraftRoute] = None) -> None:
        """Inicializa el doble.

        Args:
            route: Ruta que devolvera. Si es None se devuelve una vacia.
        """
        self.route = route or AircraftRoute()

    def lookup(self, hexcode: str, flight_callsign: str = "") -> AircraftRoute:
        """Devuelve la ruta configurada."""
        return self.route


def make_tracker(
    aircraft: List[RawAircraft],
    tmp_path: Path,
    route: Optional[AircraftRoute] = None,
    airports: Optional[pd.DataFrame] = None,
) -> FlightTracker:
    """Construye un seguidor con dobles y rutas temporales.

    Args:
        aircraft: Aviones que vera el seguidor.
        tmp_path: Directorio temporal para las fichas.
        route: Ruta que devolvera adsbdb.
        airports: Tabla de aeropuertos.

    Returns:
        El seguidor listo para usar.
    """
    config = get_config()
    config = replace(config, paths=replace(config.paths, flight_records=tmp_path))
    return FlightTracker(
        config,
        adsb=FakeAdsb(aircraft),
        adsbdb=FakeAdsbdb(route),
        airports=airports if airports is not None else pd.DataFrame(),
    )


def test_tracker_emits_a_card_for_a_landing(tmp_path: Path) -> None:
    """Un avion que toca tierra genera una ficha de llegada.

    El avion recien aterrizado lleva `gs` alta: sin coordenadas de pista
    configuradas, la velocidad es lo unico que distingue al que acaba de tomar
    del que esta aparcado.
    """
    tracker = make_tracker(
        [{"hex": "abc123", "flight": "EZY1 ", "alt_baro": 900}], tmp_path
    )
    tracker.step(now=100)
    tracker.adsb.aircraft = [
        {"hex": "abc123", "flight": "EZY1 ", "alt_baro": "ground", "gs": 120}
    ]
    cards = tracker.step(now=110)

    assert len(cards) == 1
    assert cards[0].phase is FlightPhase.ARRIVAL
    assert cards[0].callsign == "EZY1"
    assert cards[0].event_time == 110


def test_a_landing_is_caught_even_if_the_aircraft_stops_emitting(
    tmp_path: Path,
) -> None:
    """Un avion que se pierde bajando cerca del suelo se da por aterrizado.

    Es el caso habitual: al tomar tierra muchos dejan de emitir y nunca llega a
    verse la muestra con altura cero.
    """
    descending = {"hex": "abc123", "flight": "EZY1", "alt_baro": 500, "baro_rate": -700}
    tracker = make_tracker([descending], tmp_path)
    tracker.step(now=100)

    # Deja de verse: la consulta ya no lo devuelve.
    tracker.adsb.aircraft = []
    config = tracker.identification
    cards = tracker.step(now=100 + config.lost_after_seconds + 1)

    assert len(cards) == 1
    assert cards[0].phase is FlightPhase.ARRIVAL


def test_tracker_does_not_repeat_an_unchanged_card(tmp_path: Path) -> None:
    """La misma ficha no se emite dos veces seguidas."""
    aircraft = [{"hex": "abc123", "flight": "EZY1", "alt_baro": 900}]
    tracker = make_tracker(aircraft, tmp_path)
    assert len(tracker.step(now=100)) == 1
    assert tracker.step(now=110) == []


def test_tracker_writes_the_records_file(tmp_path: Path) -> None:
    """Cada ficha emitida se anade al JSONL."""
    tracker = make_tracker(
        [{"hex": "abc123", "flight": "EZY1", "alt_baro": 900}], tmp_path
    )
    tracker.step(now=100)
    path = tmp_path / RECORDS_FILENAME
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["icao24"] == "abc123"
    assert payload["callsign"] == "EZY1"


def test_tracker_ignores_high_passing_flights(tmp_path: Path) -> None:
    """Un vuelo de paso a altura de crucero no genera ficha."""
    tracker = make_tracker(
        [{"hex": "abc123", "flight": "TRA10G", "alt_baro": 31000}], tmp_path
    )
    assert tracker.step(now=100) == []


def test_tracker_treats_a_parked_aircraft_as_not_operating(tmp_path: Path) -> None:
    """Un avion en tierra y lento esta aparcado, no operando en la pista."""
    tracker = make_tracker(
        [{"hex": "abc123", "flight": "", "alt_baro": "ground", "gs": 5}], tmp_path
    )
    assert tracker.step(now=100) == []
