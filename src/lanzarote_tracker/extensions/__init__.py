"""AMPLIACIONES: NO forman parte del alcance base del proyecto.

Nada de este paquete se usa desde el codigo base: cada modulo es un stub con un
docstring que explica que hara y `raise NotImplementedError`. El orden de
prioridad acordado es:

1. :mod:`lanzarote_tracker.extensions.kalman` - filtro de Kalman.
2. :mod:`lanzarote_tracker.extensions.camera_motion` - compensacion del
   movimiento de camara con homografia.
3. :mod:`lanzarote_tracker.extensions.live` - funcionamiento en directo.
4. :mod:`lanzarote_tracker.extensions.texture_security` - seguridad reforzada
   con texturas.

Ninguna se empieza hasta que el alcance base este terminado.
"""

from __future__ import annotations

__all__ = ["kalman", "camera_motion", "live", "texture_security"]
