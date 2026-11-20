# `docs/` — documentación de diseño

Aquí van los documentos que acompañan al proyecto y que no son el informe (que está en
[`report/`](../report/README.md)).

## Qué debería acabar aquí

- **`propuesta.md`** — la propuesta de proyecto entregada al profesor: qué se va a
  hacer, con qué datos y con qué técnicas.
- **`diagrama_bloques.md`** — el diagrama de bloques en Mermaid, en una versión
  ampliable con más detalle que la del README, y la figura exportada para el informe.
- **`decisiones.md`** — notas de diseño: por qué se ha elegido cada técnica y qué
  alternativas se descartaron. Sirve para redactar la metodología del informe sin tener
  que reconstruir el razonamiento meses después.
- **`aprobaciones.md`** — qué se ha consultado al profesor y qué ha respondido. En
  particular, el estado de los backends `coco` y `cnn`, que usan *deep learning* y están
  desactivados a la espera de aprobación.

## Puntos abiertos a día de hoy

- **Backends de *deep learning***: `coco` (detección) y `cnn` (clasificación) están
  desactivados. Falta preguntar al profesor si se pueden usar.
- **OpenSky**: falta verificar en su documentación actual qué método de autenticación
  sigue vigente y cuáles son los límites de uso. El cliente deja un TODO explícito y no
  asume ningún esquema.
- **Cabecera en uso y orientación**: la tabla que relaciona ver el avión de frente o de
  espaldas con aterrizar o despegar según la cabecera (03 o 21) está puesta como
  hipótesis en `classification/arrival_departure.py`. Hay que confirmarla observando
  clips reales de las dos cabeceras antes de darla por buena.
- **Rangos HSV**: los de `config.py` son valores de partida. Hay que reajustarlos con
  `scripts/tune_hsv.py` sobre las tarjetas **ya impresas** y con la iluminación real.
