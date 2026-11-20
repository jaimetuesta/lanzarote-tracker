"""AMPLIACION 3 (NO BASE): funcionamiento en directo sobre el stream.

Que hara: ejecutar el sistema completo sobre el directo en vez de sobre clips
grabados. El sistema base trabaja SIEMPRE sobre clips, porque:

- permite repetir el mismo material tantas veces como haga falta mientras se
  ajustan los parametros;
- las consultas de llegadas y salidas de OpenSky solo estan disponibles al dia
  siguiente, asi que en directo la ficha quedaria incompleta;
- en directo no se puede volver atras cuando algo falla.

Problemas que habra que resolver: descartar fotogramas cuando el procesamiento
no siga el ritmo del stream, reconectar cuando el directo se corte, y decidir
que mostrar mientras la identificacion todavia no ha respondido.
"""

from __future__ import annotations

from typing import Optional


class LivePipeline:
    """Ejecutara el sistema completo sobre el directo, sin grabar clips."""

    def __init__(self, stream_url: Optional[str] = None) -> None:
        """Preparara el pipeline en directo.

        Args:
            stream_url: URL del directo. Si es None se toma de la configuracion.

        Raises:
            NotImplementedError: Siempre: es una ampliacion, no alcance base.
        """
        raise NotImplementedError(
            "AMPLIACION no implementada: funcionamiento en directo "
            "(extensions/live.py). El sistema base trabaja sobre clips grabados."
        )

    def run(self) -> None:
        """Ejecutara el bucle en directo hasta que el usuario lo pare.

        Raises:
            NotImplementedError: Siempre: es una ampliacion.
        """
        raise NotImplementedError(
            "AMPLIACION no implementada: funcionamiento en directo (extensions/live.py)."
        )


__all__ = ["LivePipeline"]
