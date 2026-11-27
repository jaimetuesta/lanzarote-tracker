"""Tests de la maquina de estados del desbloqueo.

Estos tests se escribieron CONTRA LA INTERFAZ de
:class:`~lanzarote_tracker.security.sequence_fsm.SequenceFsm` antes de
implementarla, y estuvieron marcados como `xfail` mientras tanto. La maquina ya
esta implementada y el marcador se ha retirado: desde aqui protegen de
regresiones de verdad.

La maquina es logica pura (no toca OpenCV ni la camara) y recibe el tiempo
desde fuera en `FrameObservation.timestamp`, de modo que estos tests controlan
el reloj y no necesitan esperas reales.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import pytest

from lanzarote_tracker.config import SecurityConfig, get_config
from lanzarote_tracker.security.sequence_fsm import FrameObservation, SequenceFsm
from lanzarote_tracker.security.types import Card, Color, SecurityState, Shape

# --- Utilidades de los tests ------------------------------------------------
def make_card(shape: str, color: str) -> Card:
    """Construye una tarjeta a partir de los nombres de forma y color.

    Args:
        shape: Nombre de la forma, p. ej. "circle".
        color: Nombre del color, p. ej. "red".

    Returns:
        La tarjeta, sin datos geometricos, que la maquina no usa.
    """
    return Card(shape=Shape(shape), color=Color(color))


def feed(
    fsm: SequenceFsm,
    card: Optional[Card],
    timestamp: float,
    card_visible: Optional[bool] = None,
) -> SecurityState:
    """Entrega un fotograma a la maquina.

    Args:
        fsm: Maquina de estados.
        card: Tarjeta confirmada en este fotograma, o None.
        timestamp: Marca de tiempo del fotograma, en segundos.
        card_visible: Si se ve una tarjeta. Por defecto, True cuando hay
            tarjeta confirmada.

    Returns:
        El estado en el que queda la maquina.
    """
    visible = card is not None if card_visible is None else card_visible
    return fsm.update(
        FrameObservation(timestamp=timestamp, card_visible=visible, confirmed_card=card)
    )


def feed_gap(fsm: SequenceFsm, start: float, frames: int) -> float:
    """Entrega varios fotogramas SIN tarjeta, que es el hueco entre lecturas.

    Args:
        fsm: Maquina de estados.
        start: Marca de tiempo del primer fotograma del hueco.
        frames: Numero de fotogramas sin tarjeta.

    Returns:
        La marca de tiempo del siguiente fotograma libre.
    """
    timestamp = start
    for _ in range(frames):
        feed(fsm, None, timestamp, card_visible=False)
        timestamp += 0.04
    return timestamp


def feed_sequence(
    fsm: SequenceFsm,
    cards: List[Card],
    config: SecurityConfig,
    start: float = 0.0,
) -> float:
    """Entrega una secuencia completa de tarjetas, con su hueco entre cada una.

    Args:
        fsm: Maquina de estados.
        cards: Tarjetas a entregar, en orden.
        config: Configuracion de seguridad, de la que se toma `gap_frames`.
        start: Marca de tiempo del primer fotograma.

    Returns:
        La marca de tiempo del siguiente fotograma libre.
    """
    timestamp = start
    for card in cards:
        timestamp = feed_gap(fsm, timestamp, config.gap_frames)
        feed(fsm, card, timestamp)
        timestamp += 0.04
    return timestamp


@pytest.fixture()
def config() -> SecurityConfig:
    """Configuracion de seguridad por defecto."""
    return get_config().security


@pytest.fixture()
def fsm(config: SecurityConfig) -> SequenceFsm:
    """Maquina de estados recien creada."""
    return SequenceFsm(config)


@pytest.fixture()
def correct_cards(config: SecurityConfig) -> List[Card]:
    """Las tarjetas de la secuencia correcta."""
    return [make_card(shape, color) for shape, color in config.expected_sequence]


# --- Estado inicial ---------------------------------------------------------
def test_starts_waiting(fsm: SequenceFsm) -> None:
    """La maquina arranca Esperando, sin tarjetas leidas ni fallos."""
    assert fsm.state is SecurityState.WAITING
    assert fsm.read_cards == ()
    assert fsm.failed_attempts == 0
    assert fsm.is_unlocked is False


def test_empty_frames_keep_it_waiting(fsm: SequenceFsm) -> None:
    """Sin tarjetas a la vista, la maquina sigue Esperando."""
    feed_gap(fsm, 0.0, 30)
    assert fsm.state is SecurityState.WAITING


# --- Secuencia correcta -----------------------------------------------------
def test_correct_sequence_unlocks(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """La secuencia correcta desbloquea el sistema."""
    feed_sequence(fsm, correct_cards, config)
    assert fsm.state is SecurityState.UNLOCKED
    assert fsm.is_unlocked is True


def test_reads_accumulate_one_by_one(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Cada tarjeta confirmada se va acumulando en orden."""
    timestamp = 0.0
    for index, card in enumerate(correct_cards[:-1], start=1):
        timestamp = feed_gap(fsm, timestamp, config.gap_frames)
        feed(fsm, card, timestamp)
        timestamp += 0.04
        assert len(fsm.read_cards) == index
        assert fsm.state is not SecurityState.UNLOCKED


def test_unlocked_is_terminal_until_reset(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Una vez desbloqueado, nuevos fotogramas no cambian el estado."""
    timestamp = feed_sequence(fsm, correct_cards, config)
    feed(fsm, make_card("square", "red"), timestamp + 1.0)
    assert fsm.state is SecurityState.UNLOCKED


# --- Secuencia incorrecta ---------------------------------------------------
def test_wrong_sequence_goes_to_error(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Una secuencia incorrecta termina en Error y suma un fallo."""
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    feed_sequence(fsm, wrong, config)
    assert fsm.is_unlocked is False
    assert fsm.failed_attempts == 1


def test_sequence_is_checked_only_at_the_end(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Una primera tarjeta equivocada NO aborta el intento al instante.

    La secuencia se comprueba entera al final, para no filtrar que posicion
    estaba mal.
    """
    wrong = list(correct_cards)
    wrong[0] = make_card("pentagon", "red")
    timestamp = feed_gap(fsm, 0.0, config.gap_frames)
    feed(fsm, wrong[0], timestamp)
    assert fsm.state is not SecurityState.ERROR
    assert fsm.failed_attempts == 0


def test_right_cards_in_wrong_order_fail(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Las tarjetas correctas en orden equivocado no desbloquean."""
    feed_sequence(fsm, list(reversed(correct_cards)), config)
    assert fsm.is_unlocked is False


# --- Hueco entre lecturas ---------------------------------------------------
def test_consecutive_reads_without_gap_are_ignored(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Sin fotogramas sin tarjeta de por medio, la segunda lectura se ignora.

    Si no, mantener una tarjeta quieta delante de la camara contaria como
    varias lecturas seguidas.
    """
    timestamp = feed_gap(fsm, 0.0, config.gap_frames)
    feed(fsm, correct_cards[0], timestamp)
    feed(fsm, correct_cards[1], timestamp + 0.04)
    assert len(fsm.read_cards) == 1


# --- Tiempo agotado ---------------------------------------------------------
def test_timeout_without_card_aborts_the_attempt(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Pasado el tiempo maximo sin tarjeta, el intento en curso se aborta."""
    timestamp = feed_gap(fsm, 0.0, config.gap_frames)
    feed(fsm, correct_cards[0], timestamp)
    feed(fsm, None, timestamp + config.max_seconds_without_card + 1.0, card_visible=False)
    assert fsm.read_cards == ()


# --- Bloqueo ----------------------------------------------------------------
def test_three_failures_lock_the_system(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Tras `max_failed_attempts` fallos seguidos el sistema se bloquea."""
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    timestamp = 0.0
    for _ in range(config.max_failed_attempts):
        timestamp = feed_sequence(fsm, wrong, config, start=timestamp)
    assert fsm.state is SecurityState.LOCKED
    assert fsm.is_locked is True


def test_locked_ignores_cards(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Durante el bloqueo no se acepta ninguna tarjeta, ni las correctas."""
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    timestamp = 0.0
    for _ in range(config.max_failed_attempts):
        timestamp = feed_sequence(fsm, wrong, config, start=timestamp)
    feed_sequence(fsm, correct_cards, config, start=timestamp)
    assert fsm.is_unlocked is False


def test_lockout_expires_after_its_duration(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Pasados `lockout_seconds` el sistema vuelve a aceptar tarjetas."""
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    timestamp = 0.0
    for _ in range(config.max_failed_attempts):
        timestamp = feed_sequence(fsm, wrong, config, start=timestamp)

    after_lockout = timestamp + config.lockout_seconds + 1.0
    feed(fsm, None, after_lockout, card_visible=False)
    assert fsm.state is SecurityState.WAITING
    assert fsm.failed_attempts == 0

    feed_sequence(fsm, correct_cards, config, start=after_lockout + 0.04)
    assert fsm.is_unlocked is True


def test_a_success_clears_previous_failures(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Un intento acertado no deja fallos pendientes para el siguiente."""
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    timestamp = feed_sequence(fsm, wrong, config)
    assert fsm.failed_attempts == 1
    feed_sequence(fsm, correct_cards, config, start=timestamp)
    assert fsm.is_unlocked is True
    assert fsm.failed_attempts == 0


# --- Reinicio ---------------------------------------------------------------
def test_reset_clears_the_attempt_but_keeps_failures(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """reset() aborta el intento pero conserva los fallos acumulados."""
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    feed_sequence(fsm, wrong, config)
    fsm.reset()
    assert fsm.state is SecurityState.WAITING
    assert fsm.read_cards == ()
    assert fsm.failed_attempts == 1


def test_reset_can_clear_failures(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """reset(clear_failures=True) tambien pone los fallos a cero."""
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    feed_sequence(fsm, wrong, config)
    fsm.reset(clear_failures=True)
    assert fsm.failed_attempts == 0


# --- Configuracion alternativa ----------------------------------------------
def test_respects_a_shorter_sequence(config: SecurityConfig) -> None:
    """La longitud de la secuencia se toma de la configuracion, no esta fijada."""
    short = replace(
        config,
        sequence_length=2,
        expected_sequence=(("circle", "red"), ("square", "blue")),
    )
    fsm = SequenceFsm(short)
    feed_sequence(fsm, [make_card("circle", "red"), make_card("square", "blue")], short)
    assert fsm.is_unlocked is True


def test_status_message_is_a_non_empty_string(fsm: SequenceFsm) -> None:
    """status_message devuelve texto para superponer en el video."""
    message = fsm.status_message()
    assert isinstance(message, str)
    assert message


# --- Decisiones de diseno que conviene dejar fijadas ------------------------
def test_timeout_does_not_count_as_a_failure(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Agotar el tiempo aborta el intento pero NO acerca el bloqueo.

    Quien se va a medio desbloquear no esta atacando nada, y como la secuencia
    solo se comprueba al final, abortar a mitad no revela ni una posicion. Si
    los tiempos agotados sumasen fallo, dejar la tarjeta a medias tres veces
    bloquearia el sistema sin motivo.
    """
    for attempt in range(3):
        start = attempt * 100.0
        timestamp = feed_gap(fsm, start, config.gap_frames)
        feed(fsm, correct_cards[0], timestamp)
        feed(
            fsm,
            None,
            timestamp + config.max_seconds_without_card + 1.0,
            card_visible=False,
        )
    assert fsm.failed_attempts == 0
    assert fsm.is_locked is False


def test_error_is_visible_for_one_frame_then_clears(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """El aviso de secuencia incorrecta dura un fotograma.

    Lo justo para que la interfaz pueda mostrarlo; despues se vuelve a
    Esperando para no dejar la pantalla clavada en el error.
    """
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    timestamp = feed_sequence(fsm, wrong, config)
    assert fsm.state is SecurityState.ERROR
    feed(fsm, None, timestamp, card_visible=False)
    assert fsm.state is SecurityState.WAITING


def test_checking_is_never_observed_from_outside(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """La comprobacion se atraviesa dentro de la misma llamada a update.

    Quien recibe el estado ve ya el resultado, no el paso intermedio.
    """
    states = []
    timestamp = 0.0
    for card in correct_cards:
        timestamp = feed_gap(fsm, timestamp, config.gap_frames)
        states.append(feed(fsm, card, timestamp))
        timestamp += 0.04
    assert SecurityState.CHECKING not in states
    assert states[-1] is SecurityState.UNLOCKED


def test_lockout_countdown_decreases(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """Los segundos restantes del bloqueo bajan con el tiempo observado."""
    wrong = list(correct_cards)
    wrong[-1] = make_card("square", "red")
    timestamp = 0.0
    for _ in range(config.max_failed_attempts):
        timestamp = feed_sequence(fsm, wrong, config, start=timestamp)

    full = fsm.seconds_left_locked
    feed(fsm, None, timestamp + 10.0, card_visible=False)
    assert fsm.seconds_left_locked < full
    assert fsm.seconds_left_locked > 0


def test_status_message_follows_the_state(
    fsm: SequenceFsm, correct_cards: List[Card], config: SecurityConfig
) -> None:
    """El mensaje cambia con el estado y dice por que tarjeta se va."""
    assert "1" in fsm.status_message()
    timestamp = feed_gap(fsm, 0.0, config.gap_frames)
    feed(fsm, correct_cards[0], timestamp)
    assert "2" in fsm.status_message()
    feed_sequence(fsm, correct_cards[1:], config, start=timestamp + 0.04)
    assert fsm.status_message() == "Desbloqueado"


def test_the_first_card_needs_no_previous_gap(
    fsm: SequenceFsm, correct_cards: List[Card]
) -> None:
    """Nada mas arrancar se acepta una tarjeta sin esperar fotogramas vacios.

    Exigir un hueco que nadie ha visto todavia solo retrasaria el primer
    desbloqueo sin aportar nada.
    """
    feed(fsm, correct_cards[0], 0.0)
    assert len(fsm.read_cards) == 1
