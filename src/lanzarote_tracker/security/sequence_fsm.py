"""Maquina de estados del desbloqueo por secuencia de tarjetas.

Estados (ver :class:`~lanzarote_tracker.security.types.SecurityState`):

    WAITING  -> Esperando: no hay lectura en curso.
    READING  -> Leyendo k: acumulando la tarjeta numero k.
    CHECKING -> Comprobacion: hay 4 tarjetas y se compara la secuencia entera.
    UNLOCKED -> Desbloqueado.
    ERROR    -> Error: secuencia incorrecta o tiempo agotado.
    LOCKED   -> Bloqueo: 3 fallos seguidos, 30 s sin aceptar tarjetas.

Reglas, todas parametrizadas en :class:`~lanzarote_tracker.config.SecurityConfig`:

- Una tarjeta se da por leida tras N fotogramas consecutivos con la misma
  lectura. De eso se encarga
  :class:`~lanzarote_tracker.security.card_reader.StableCardReader`, que entrega
  aqui la tarjeta ya confirmada.
- Entre dos lecturas tiene que haber fotogramas SIN tarjeta
  (`gap_frames`), para que mantener una tarjeta quieta delante de la camara no
  cuente como varias lecturas seguidas.
- La secuencia se comprueba ENTERA al final, nunca tarjeta a tarjeta: asi un
  atacante no puede averiguar la secuencia posicion a posicion.
- Si pasan mas de `max_seconds_without_card` segundos sin ver ninguna tarjeta
  con una lectura en curso, el intento se aborta (ERROR).
- Tras `max_failed_attempts` fallos consecutivos se entra en LOCKED durante
  `lockout_seconds` segundos.

La maquina es LOGICA PURA: no toca OpenCV ni la camara, recibe observaciones ya
digeridas. Por eso se puede probar entera con pytest sin camara ni imagenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from lanzarote_tracker.config import SecurityConfig
from lanzarote_tracker.security.types import Card, SecurityState


@dataclass(frozen=True)
class FrameObservation:
    """Lo que la maquina de estados sabe de un fotograma.

    Attributes:
        timestamp: Marca de tiempo monotona en segundos, normalmente de
            `time.monotonic()`. La maquina NO llama al reloj por su cuenta,
            para que los tests puedan controlar el tiempo.
        card_visible: Si en el fotograma se esta viendo alguna tarjeta, este
            confirmada o no. Sirve para exigir el hueco entre lecturas y para
            el tiempo maximo sin tarjeta.
        confirmed_card: Tarjeta confirmada en ESTE fotograma, es decir, la que
            acaba de completar sus N fotogramas consecutivos. None en todos los
            demas fotogramas.
    """

    timestamp: float
    card_visible: bool = False
    confirmed_card: Optional[Card] = None


class SequenceFsm:
    """Maquina de estados del desbloqueo por secuencia de 4 tarjetas.

    Uso tipico, un fotograma por iteracion::

        fsm = SequenceFsm(config.security)
        state = fsm.update(FrameObservation(
            timestamp=time.monotonic(),
            card_visible=reader.current_card is not None,
            confirmed_card=reader.update(frame),
        ))
        if fsm.is_unlocked:
            ...

    Attributes:
        config: Configuracion de seguridad con N, tiempos y secuencia esperada.
    """

    def __init__(self, config: SecurityConfig) -> None:
        """Inicializa la maquina en el estado WAITING, sin fallos acumulados.

        Args:
            config: Configuracion de seguridad.
        """
        raise NotImplementedError(
            "Pendiente: SequenceFsm.__init__ (security/sequence_fsm.py)."
        )

    # --- Estado observable ---------------------------------------------------
    @property
    def state(self) -> SecurityState:
        """Estado actual de la maquina."""
        raise NotImplementedError(
            "Pendiente: SequenceFsm.state (security/sequence_fsm.py)."
        )

    @property
    def read_cards(self) -> Tuple[Card, ...]:
        """Tarjetas confirmadas en el intento en curso, en orden de lectura."""
        raise NotImplementedError(
            "Pendiente: SequenceFsm.read_cards (security/sequence_fsm.py)."
        )

    @property
    def current_index(self) -> int:
        """Indice k de la tarjeta que se esta leyendo, empezando en 0."""
        raise NotImplementedError(
            "Pendiente: SequenceFsm.current_index (security/sequence_fsm.py)."
        )

    @property
    def failed_attempts(self) -> int:
        """Intentos fallidos consecutivos acumulados."""
        raise NotImplementedError(
            "Pendiente: SequenceFsm.failed_attempts (security/sequence_fsm.py)."
        )

    @property
    def is_unlocked(self) -> bool:
        """True si la maquina esta en UNLOCKED."""
        raise NotImplementedError(
            "Pendiente: SequenceFsm.is_unlocked (security/sequence_fsm.py)."
        )

    @property
    def is_locked(self) -> bool:
        """True si la maquina esta en LOCKED y el bloqueo sigue vigente."""
        raise NotImplementedError(
            "Pendiente: SequenceFsm.is_locked (security/sequence_fsm.py)."
        )

    # --- Transiciones --------------------------------------------------------
    def update(self, observation: FrameObservation) -> SecurityState:
        """Avanza la maquina un fotograma y devuelve el estado resultante.

        Transiciones, en el orden en que se evaluan:

        1. En LOCKED: si no han pasado `lockout_seconds` desde el bloqueo, se
           ignora todo y se sigue en LOCKED; si han pasado, se pasa a WAITING y
           se ponen a cero los fallos.
        2. En UNLOCKED: estado terminal, solo sale con :meth:`reset`.
        3. Tiempo agotado: con una lectura en curso y mas de
           `max_seconds_without_card` segundos sin ver tarjeta, se va a ERROR.
        4. Hueco entre lecturas: una tarjeta confirmada solo se acepta si antes
           se han visto al menos `gap_frames` fotogramas sin tarjeta; si no, se
           ignora.
        5. Tarjeta confirmada: se anade a `read_cards`. Si ya hay
           `sequence_length`, se pasa a CHECKING y se compara la secuencia
           ENTERA con `config.expected_sequence`: si coincide, UNLOCKED; si no,
           ERROR con un fallo mas.
        6. ERROR: se limpia el intento y se vuelve a WAITING, salvo que los
           fallos hayan llegado a `max_failed_attempts`, en cuyo caso se pasa a
           LOCKED.

        Args:
            observation: Lo observado en este fotograma.

        Returns:
            El estado en el que queda la maquina tras procesar el fotograma.
        """
        raise NotImplementedError(
            "Pendiente: SequenceFsm.update (security/sequence_fsm.py). "
            "Es el nucleo del bloque de seguridad: ver las transiciones del docstring."
        )

    def reset(self, clear_failures: bool = False) -> None:
        """Aborta el intento en curso y vuelve a WAITING.

        Args:
            clear_failures: Si tambien se ponen a cero los fallos acumulados.
                False tras un fallo (los fallos deben acumularse hasta el
                bloqueo) y True al arrancar el sistema o tras un desbloqueo.
        """
        raise NotImplementedError(
            "Pendiente: SequenceFsm.reset (security/sequence_fsm.py)."
        )

    def status_message(self) -> str:
        """Devuelve un texto breve del estado, para superponerlo en el video.

        Returns:
            Un mensaje como "Esperando tarjeta 2 de 4" o
            "Bloqueado: 23 s restantes".
        """
        raise NotImplementedError(
            "Pendiente: SequenceFsm.status_message (security/sequence_fsm.py)."
        )


__all__ = ["FrameObservation", "SequenceFsm"]
