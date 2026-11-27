"""Tests de integracion del bloque de seguridad: lector + maquina de estados.

POR QUE EXISTE ESTE FICHERO. El lector y la maquina de estados pasaban cada uno
sus tests y aun asi el desbloqueo no avanzaba de "Esperando tarjeta 1 de 4": NO
SE PODIA ACEPTAR NINGUNA TARJETA.

El motivo era la costura entre los dos. La maquina exige fotogramas sin tarjeta
entre lecturas, y los contaba en el instante de recibir la confirmacion. Pero
el lector necesita N fotogramas con la tarjeta A LA VISTA para confirmarla, asi
que en ese instante el hueco vale siempre 0. Los tests de la maquina entregaban
la tarjeta confirmada justo despues del hueco, una secuencia que en la realidad
no ocurre nunca.

De ahi que estos tests conduzcan la maquina con el LECTOR DE VERDAD sobre
imagenes de tarjetas reales, en vez de con observaciones a mano. Es lo unico
que puede pillar un fallo que vive entre dos componentes y no dentro de ninguno.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytest

from lanzarote_tracker.config import SecurityConfig, get_config
from lanzarote_tracker.security.card_reader import StableCardReader
from lanzarote_tracker.security.sequence_fsm import FrameObservation, SequenceFsm
from lanzarote_tracker.security.types import SecurityState


@pytest.fixture()
def config() -> SecurityConfig:
    """Configuracion de seguridad por defecto."""
    return get_config().security


def card_image(shape: str, color: str) -> np.ndarray:
    """Carga la imagen de una tarjeta generada.

    Args:
        shape: Nombre de la forma.
        color: Nombre del color.

    Returns:
        La imagen BGR de la tarjeta.
    """
    path = Path("assets/cards") / f"{shape}_{color}.png"
    image = cv2.imread(str(path))
    assert image is not None, f"falta {path}"
    return image


def blank_image() -> np.ndarray:
    """Devuelve un fotograma sin ninguna tarjeta.

    Returns:
        Una imagen blanca del tamano de las tarjetas.
    """
    return np.full((700, 700, 3), 255, dtype=np.uint8)


class Session:
    """Conduce el lector y la maquina de estados como lo hace el script real.

    Attributes:
        reader: Lector de tarjetas.
        fsm: Maquina de estados.
    """

    def __init__(self, config: SecurityConfig) -> None:
        """Inicializa la sesion.

        Args:
            config: Configuracion de seguridad.
        """
        self.config = config
        self.reader = StableCardReader(config)
        self.fsm = SequenceFsm(config)
        self._time = 0.0
        self._step = 1.0 / 30.0

    def feed(self, image: np.ndarray, frames: int) -> List[SecurityState]:
        """Entrega el mismo fotograma varias veces, como una camara real.

        Args:
            image: Fotograma a repetir.
            frames: Numero de repeticiones.

        Returns:
            Los estados por los que ha pasado la maquina.
        """
        states = []
        for _ in range(frames):
            confirmed = self.reader.update(image)
            states.append(
                self.fsm.update(
                    FrameObservation(
                        timestamp=self._time,
                        card_visible=self.reader.current_card is not None,
                        confirmed_card=confirmed,
                    )
                )
            )
            self._time += self._step
        return states

    def show(self, shape: str, color: str, extra: int = 2) -> None:
        """Retira la tarjeta anterior y ensena una nueva hasta confirmarla.

        Args:
            shape: Forma de la tarjeta.
            color: Color de la tarjeta.
            extra: Fotogramas de mas, por encima de los imprescindibles.
        """
        self.feed(blank_image(), self.config.gap_frames + extra)
        self.feed(card_image(shape, color), self.config.consecutive_frames + extra)

    def show_sequence(self, sequence: Tuple[Tuple[str, str], ...]) -> None:
        """Ensena una secuencia entera de tarjetas.

        Args:
            sequence: Pares (forma, color) en orden.
        """
        for shape, color in sequence:
            self.show(shape, color)


# --- El caso que fallaba ----------------------------------------------------
def test_showing_a_card_registers_a_reading(config: SecurityConfig) -> None:
    """Ensenar una tarjeta hace avanzar el contador.

    Es el test que faltaba: era justo lo que no ocurria. La maquina se quedaba
    en "Esperando tarjeta 1 de 4" para siempre.
    """
    session = Session(config)
    session.show("circle", "red")
    assert len(session.fsm.read_cards) == 1, (
        "la tarjeta se ha confirmado pero la maquina no la ha aceptado"
    )


def test_the_correct_sequence_unlocks(config: SecurityConfig) -> None:
    """La secuencia configurada desbloquea de principio a fin."""
    session = Session(config)
    session.show_sequence(config.expected_sequence)
    assert session.fsm.is_unlocked is True


def test_the_cards_are_read_in_order(config: SecurityConfig) -> None:
    """Cada tarjeta se acepta en su turno, no de golpe al final."""
    session = Session(config)
    for index, (shape, color) in enumerate(config.expected_sequence[:-1], start=1):
        session.show(shape, color)
        assert len(session.fsm.read_cards) == index


# --- La regla del hueco, con el lector de verdad -----------------------------
def test_holding_one_card_does_not_fill_the_sequence(
    config: SecurityConfig,
) -> None:
    """Sostener una sola tarjeta no completa la secuencia entera.

    Es la razon de ser de la regla del hueco: si cada N fotogramas sostenidos
    contasen como una lectura nueva, bastaria una tarjeta para desbloquear.
    """
    session = Session(config)
    session.feed(blank_image(), config.gap_frames + 2)
    session.feed(card_image("circle", "red"), config.consecutive_frames * 5)
    assert len(session.fsm.read_cards) == 1


def test_a_card_without_a_gap_is_ignored(config: SecurityConfig) -> None:
    """Cambiar de tarjeta sin retirarla primero no cuenta como lectura nueva."""
    session = Session(config)
    session.show("circle", "red")
    # Se cambia de tarjeta sin pasar por un fotograma vacio.
    session.feed(card_image("triangle", "green"), config.consecutive_frames + 2)
    assert len(session.fsm.read_cards) == 1


def test_a_gap_allows_the_next_card(config: SecurityConfig) -> None:
    """Retirando la tarjeta, la siguiente si se acepta."""
    session = Session(config)
    session.show("circle", "red")
    session.show("triangle", "green")
    assert len(session.fsm.read_cards) == 2


# --- Secuencias incorrectas --------------------------------------------------
def test_a_wrong_sequence_does_not_unlock(config: SecurityConfig) -> None:
    """Una secuencia equivocada no desbloquea y suma un fallo."""
    wrong = list(config.expected_sequence)
    wrong[-1] = ("square", "red")
    session = Session(config)
    session.show_sequence(tuple(wrong))
    assert session.fsm.is_unlocked is False
    assert session.fsm.failed_attempts == 1


def test_the_right_cards_in_the_wrong_order_do_not_unlock(
    config: SecurityConfig,
) -> None:
    """El orden importa: las mismas cuatro tarjetas al reves no valen."""
    session = Session(config)
    session.show_sequence(tuple(reversed(config.expected_sequence)))
    assert session.fsm.is_unlocked is False


def test_three_wrong_attempts_lock_the_system(config: SecurityConfig) -> None:
    """Tres secuencias incorrectas seguidas bloquean el sistema."""
    wrong = list(config.expected_sequence)
    wrong[-1] = ("square", "red")
    session = Session(config)
    for _ in range(config.max_failed_attempts):
        session.show_sequence(tuple(wrong))
    assert session.fsm.state is SecurityState.LOCKED


def test_a_retry_after_a_failure_can_unlock(config: SecurityConfig) -> None:
    """Tras un fallo se puede volver a intentar y acertar."""
    wrong = list(config.expected_sequence)
    wrong[-1] = ("square", "red")
    session = Session(config)
    session.show_sequence(tuple(wrong))
    assert session.fsm.is_unlocked is False
    session.show_sequence(config.expected_sequence)
    assert session.fsm.is_unlocked is True
    assert session.fsm.failed_attempts == 0


# --- Robustez del lector dentro del bucle ------------------------------------
def test_a_flicker_does_not_confirm_a_card(config: SecurityConfig) -> None:
    """Una tarjeta que aparece y desaparece no llega a confirmarse.

    Es lo que protege de un reflejo o de un fotograma movido.
    """
    session = Session(config)
    session.feed(blank_image(), config.gap_frames + 2)
    for _ in range(6):
        session.feed(card_image("circle", "red"), config.consecutive_frames // 3)
        session.feed(blank_image(), 2)
    assert session.fsm.read_cards == ()


def test_an_unrecognised_shape_is_not_a_reading(config: SecurityConfig) -> None:
    """Un rectangulo de color no cuenta como tarjeta."""
    session = Session(config)
    frame = np.full((700, 700, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (40, 300), (660, 420), (0, 0, 255), thickness=-1)
    session.feed(blank_image(), config.gap_frames + 2)
    session.feed(frame, config.consecutive_frames * 2)
    assert session.fsm.read_cards == ()


def test_a_shorter_sequence_also_works(config: SecurityConfig) -> None:
    """La longitud de la secuencia sale de la configuracion, no esta fijada."""
    short = replace(
        config,
        sequence_length=2,
        expected_sequence=(("square", "blue"), ("circle", "red")),
    )
    session = Session(short)
    session.show_sequence(short.expected_sequence)
    assert session.fsm.is_unlocked is True
