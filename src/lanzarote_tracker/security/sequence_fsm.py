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
  con una lectura en curso, el intento se aborta.
- Tras `max_failed_attempts` fallos consecutivos se entra en LOCKED durante
  `lockout_seconds` segundos.

La maquina es LOGICA PURA: no toca OpenCV ni la camara, recibe observaciones ya
digeridas. Por eso se puede probar entera con pytest sin camara ni imagenes.

DOS DECISIONES que conviene tener presentes al leer el codigo:

1. **Agotar el tiempo NO cuenta como fallo.** Aborta el intento y devuelve a
   Esperando, pero no acerca el bloqueo. Quien se va a medio desbloquear no
   esta atacando nada, y como la secuencia solo se comprueba al final, abortar
   a mitad no revela ni una posicion: no hay nada que un atacante pueda
   aprender repitiendolo. Solo suma fallo la secuencia COMPLETA e incorrecta.
2. **CHECKING no se llega a observar desde fuera.** Se atraviesa dentro de la
   misma llamada a :meth:`SequenceFsm.update` que recibe la ultima tarjeta, que
   devuelve ya UNLOCKED o el resultado del fallo. Existe como estado porque es
   un paso real del diagrama de bloques del informe. ERROR si es observable
   durante un fotograma, para que la interfaz pueda avisar de que la secuencia
   era incorrecta, y se limpia en la observacion siguiente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
        self.config: SecurityConfig = config
        self._state: SecurityState = SecurityState.WAITING
        self._read_cards: List[Card] = []
        self._failed_attempts: int = 0
        # Se arranca como si ya hubiera habido hueco: la primera tarjeta no
        # tiene por que esperar a que pasen fotogramas vacios que nadie ha visto.
        self._empty_frames: int = config.gap_frames
        self._last_card_seen: Optional[float] = None
        self._lockout_until: Optional[float] = None
        self._now: float = 0.0

    # --- Estado observable ---------------------------------------------------
    @property
    def state(self) -> SecurityState:
        """Estado actual de la maquina."""
        return self._state

    @property
    def read_cards(self) -> Tuple[Card, ...]:
        """Tarjetas confirmadas en el intento en curso, en orden de lectura."""
        return tuple(self._read_cards)

    @property
    def current_index(self) -> int:
        """Indice k de la tarjeta que se esta leyendo, empezando en 0."""
        return len(self._read_cards)

    @property
    def failed_attempts(self) -> int:
        """Intentos fallidos consecutivos acumulados."""
        return self._failed_attempts

    @property
    def is_unlocked(self) -> bool:
        """True si la maquina esta en UNLOCKED."""
        return self._state is SecurityState.UNLOCKED

    @property
    def is_locked(self) -> bool:
        """True si la maquina esta en LOCKED y el bloqueo sigue vigente.

        El bloqueo se levanta en la primera observacion posterior a su
        vencimiento, asi que mientras se llame a :meth:`update` por cada
        fotograma, este estado y el reloj van de la mano.
        """
        return self._state is SecurityState.LOCKED

    @property
    def seconds_left_locked(self) -> float:
        """Segundos que quedan de bloqueo, o 0.0 si no esta bloqueada.

        Se mide contra la ultima observacion recibida, no contra el reloj: la
        maquina no lo consulta por su cuenta.
        """
        if self._lockout_until is None or not self.is_locked:
            return 0.0
        return max(0.0, self._lockout_until - self._now)

    # --- Transiciones --------------------------------------------------------
    def update(self, observation: FrameObservation) -> SecurityState:
        """Avanza la maquina un fotograma y devuelve el estado resultante.

        Transiciones, en el orden en que se evaluan:

        1. En LOCKED: si no han pasado `lockout_seconds` desde el bloqueo, se
           ignora todo y se sigue en LOCKED; si han pasado, se pasa a WAITING y
           se ponen a cero los fallos.
        2. En UNLOCKED: estado terminal, solo sale con :meth:`reset`.
        3. ERROR se limpia: era el aviso de un fotograma y vuelve a WAITING.
        4. Tiempo agotado: con una lectura en curso y mas de
           `max_seconds_without_card` segundos sin ver tarjeta, se aborta el
           intento (sin sumar fallo).
        5. Hueco entre lecturas: una tarjeta confirmada solo se acepta si antes
           se han visto al menos `gap_frames` fotogramas sin tarjeta; si no, se
           ignora.
        6. Tarjeta confirmada: se anade a `read_cards`. Si ya hay
           `sequence_length`, se pasa a CHECKING y se compara la secuencia
           ENTERA con `config.expected_sequence`: si coincide, UNLOCKED; si no,
           ERROR con un fallo mas, o LOCKED si se agotan los intentos.

        Args:
            observation: Lo observado en este fotograma.

        Returns:
            El estado en el que queda la maquina tras procesar el fotograma.
        """
        self._now = observation.timestamp

        # 1. Bloqueo vigente: no se acepta nada hasta que venza.
        if self._state is SecurityState.LOCKED:
            if self._lockout_until is not None and self._now < self._lockout_until:
                return self._state
            self._clear_attempt()
            self._failed_attempts = 0
            self._lockout_until = None
            self._state = SecurityState.WAITING

        # 2. Desbloqueado es terminal.
        if self._state is SecurityState.UNLOCKED:
            return self._state

        # 3. El aviso de error dura un fotograma.
        if self._state is SecurityState.ERROR:
            self._state = SecurityState.WAITING

        # 4. Tiempo maximo sin ver ninguna tarjeta, con un intento empezado.
        if self._timed_out(observation):
            self._clear_attempt()
            self._state = SecurityState.WAITING

        # 5 y 6. Tarjeta confirmada, si el hueco previo lo permite.
        card = observation.confirmed_card
        if card is not None and self._empty_frames >= self.config.gap_frames:
            self._accept(card)

        self._track_visibility(observation)
        return self._state

    def _timed_out(self, observation: FrameObservation) -> bool:
        """Indica si se ha agotado el tiempo sin ver ninguna tarjeta.

        Args:
            observation: Lo observado en este fotograma.

        Returns:
            True si hay un intento empezado y lleva demasiado sin tarjeta.
        """
        if not self._read_cards or observation.card_visible:
            return False
        if self._last_card_seen is None:
            return False
        elapsed = observation.timestamp - self._last_card_seen
        return elapsed > self.config.max_seconds_without_card

    def _accept(self, card: Card) -> None:
        """Anade una tarjeta confirmada y comprueba si la secuencia esta completa.

        Args:
            card: Tarjeta confirmada.
        """
        self._read_cards.append(card)
        if len(self._read_cards) < self.config.sequence_length:
            self._state = SecurityState.READING
            return

        # Secuencia completa: se comprueba ENTERA, nunca posicion a posicion.
        self._state = SecurityState.CHECKING
        if self._matches_expected():
            self._clear_attempt()
            self._failed_attempts = 0
            self._state = SecurityState.UNLOCKED
            return

        self._clear_attempt()
        self._failed_attempts += 1
        if self._failed_attempts >= self.config.max_failed_attempts:
            self._lockout_until = self._now + self.config.lockout_seconds
            self._state = SecurityState.LOCKED
        else:
            self._state = SecurityState.ERROR

    def _matches_expected(self) -> bool:
        """Compara la secuencia leida con la esperada, entera.

        Returns:
            True si coinciden forma y color en todas las posiciones.
        """
        read = tuple(card.key for card in self._read_cards)
        expected = tuple(tuple(item) for item in self.config.expected_sequence)
        return read == expected

    def _track_visibility(self, observation: FrameObservation) -> None:
        """Actualiza los contadores de fotogramas vacios y de ultima tarjeta.

        Se hace DESPUES de decidir sobre la tarjeta de este fotograma: el hueco
        que cuenta es el que precede a la lectura, no el que la incluye.

        Args:
            observation: Lo observado en este fotograma.
        """
        if observation.card_visible:
            self._empty_frames = 0
            self._last_card_seen = observation.timestamp
        else:
            self._empty_frames += 1

    def _clear_attempt(self) -> None:
        """Vacia las tarjetas leidas y los contadores del intento en curso."""
        self._read_cards.clear()
        self._last_card_seen = None
        self._empty_frames = self.config.gap_frames

    def reset(self, clear_failures: bool = False) -> None:
        """Aborta el intento en curso y vuelve a WAITING.

        Args:
            clear_failures: Si tambien se ponen a cero los fallos acumulados.
                False tras un fallo (los fallos deben acumularse hasta el
                bloqueo) y True al arrancar el sistema o tras un desbloqueo.
        """
        self._clear_attempt()
        self._state = SecurityState.WAITING
        if clear_failures:
            self._failed_attempts = 0
            self._lockout_until = None

    def status_message(self) -> str:
        """Devuelve un texto breve del estado, para superponerlo en el video.

        Returns:
            Un mensaje como "Esperando tarjeta 2 de 4" o
            "Bloqueado: 23 s restantes".
        """
        total = self.config.sequence_length
        if self._state is SecurityState.LOCKED:
            return f"Bloqueado: {self.seconds_left_locked:.0f} s restantes"
        if self._state is SecurityState.UNLOCKED:
            return "Desbloqueado"
        if self._state is SecurityState.ERROR:
            remaining = self.config.max_failed_attempts - self._failed_attempts
            return f"Secuencia incorrecta. Intentos restantes: {remaining}"
        if self._state is SecurityState.CHECKING:
            return "Comprobando la secuencia..."
        return f"Esperando tarjeta {self.current_index + 1} de {total}"


__all__ = ["FrameObservation", "SequenceFsm"]
