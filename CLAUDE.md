# CLAUDE.md — contexto del proyecto LanzaroteTracker

Contexto para futuras sesiones de Claude Code en este repositorio. Léelo antes de
proponer cambios.

## Qué es esto

Proyecto final de **Visión por Ordenador I**, 3.º Ingeniería Matemática e IA, Comillas
ICAI, curso 2026/27. Autores: Javier Martínez Montelongo y Jaime Tuesta de Simón.

El sistema trabaja sobre la cámara en directo de Flightradar24 del aeropuerto de
Lanzarote (ACE/GCRR) en YouTube, una cámara que **gira y hace zoom**. Detecta y sigue al
avión, clasifica si se ve de frente o de espaldas, deduce si aterriza o despega
combinándolo con el sentido de uso de la pista, identifica el vuelo con OpenSky y
superpone una ficha con datos fijos. Antes, el usuario se desbloquea mostrando a su
webcam una secuencia de 4 tarjetas (forma + color).

El sistema base trabaja sobre **clips grabados**, no en directo. La webcam sí funciona
en vivo en el bloque de seguridad, y es la cámara que se calibra.

## Alcance base (lo único que se implementa primero)

1. Desbloqueo con secuencia de 4 tarjetas.
2. Grabación de clips del stream con marca de tiempo y extracción de fotogramas.
3. Detección del avión con recuadro en cada fotograma.
4. Seguimiento entre fotogramas pese al movimiento de la cámara.
5. Clasificación frente/espaldas → llegada o salida.
6. Identificación del vuelo con OpenSky.
7. Ficha con datos fijos superpuesta.

## Ampliaciones (`src/lanzarote_tracker/extensions/`) — NO base

Por orden de prioridad: filtro de Kalman, compensación del movimiento de cámara con
homografía, funcionamiento en directo, seguridad reforzada con texturas.

**Regla:** son solo stubs con docstring y `NotImplementedError`. **No mezcles nada de
ampliaciones en el código base** y no empieces ninguna hasta que el alcance base esté
terminado.

## Decisiones técnicas ya tomadas

No las reabras sin que te lo pidan.

- **Seguridad, por fotograma:** corrección de distorsión con la calibración de la webcam
  → HSV y umbral por rango de tono → apertura y cierre morfológicos → contorno de mayor
  área con área mínima → forma por número de vértices de `approxPolyDP`; el círculo se
  distingue por circularidad `C = 4πA/P²`.
- **El rojo necesita DOS rangos HSV unidos**, porque su tono está partido por el extremo
  del círculo cromático. En OpenCV **H va de 0 a 179**, no de 0 a 359 ni a 255.
- **Validación temporal:** una tarjeta se da por leída si se reconoce igual durante N
  fotogramas consecutivos (N = 15 inicial); debe haber fotogramas sin tarjeta entre
  lecturas; la secuencia se comprueba **entera al final**, nunca tarjeta a tarjeta;
  máximo 10 s sin tarjeta; bloqueo de 30 s tras 3 fallos seguidos. Es una máquina de
  estados con Esperando, Leyendo k, Comprobación, Desbloqueado, Error y Bloqueo.
- **La cámara del aeropuerto se mueve**, así que **NO se usa sustracción de fondo**:
  detección periódica cada K fotogramas + seguimiento entre detecciones (flujo óptico
  Lucas-Kanade con puntos Shi-Tomasi dentro de la caja, o meanShift sobre histograma de
  tono; ambos de la Práctica 5).
- **Detección del avión:** dos backends tras una misma interfaz. `classic` (segmentación
  propia por color y morfología) es el **backend por defecto**; `coco` (detector
  preentrenado, clase `airplane`) es opcional (ver "Deep learning" más abajo).
- **Clasificación frente/espaldas:** dos backends tras una interfaz. `bow` (bolsa de
  palabras SIFT + K-means + SVM, Práctica 5 apartado A) es la **línea base**; `cnn` (CNN
  propia) es opcional (ver "Deep learning" más abajo).
- **Captura:** URL del stream con yt-dlp y lectura con `cv2.VideoCapture`; los clips se
  guardan con `cv2.VideoWriter` y un JSON de metadatos con la marca de tiempo UTC de
  inicio.
- **Identificación del vuelo (cambiado el 27/11/2026).** La fuente primaria **ya no es
  OpenSky**, es **ADS-B**: `adsb.lol` y `adsb.fi` para los aviones que hay ahora sobre la
  pista, y `adsbdb.com` para modelo, matrícula, aerolínea y ruta. El motivo es que las
  consultas de llegadas y salidas de OpenSky **solo están disponibles al día siguiente**,
  así que no sirven para rellenar la ficha mientras se graba; ADS-B responde al momento y
  no pide credenciales. Lo implementó Jaime y se integró tal cual.
  OpenSky se conserva como **fuente secundaria** para confirmar a posteriori. Su
  autenticación es **OAuth2 `client_credentials`** (no usuario y contraseña): ya está
  implementada y lee `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET` del entorno. **Nunca
  metas un `credentials.json` en el repo**: está en `.gitignore` por esto.
- **El seguimiento de vuelos hay que ejecutarlo A LA VEZ que se graban los clips**
  (`scripts/track_flights.py` junto a `scripts/record_clips.py`). Los datos ADS-B son
  efímeros: un clip grabado sin el seguimiento en marcha se queda sin ficha y ya no hay
  forma de recuperarla.
- **Llegada o salida se deduce por DOS vías independientes y no hay que mezclarlas:**
  `classification/arrival_departure.py` lo hace **por visión** (frente/espaldas + sentido
  de la pista) y es lo que evalúa el proyecto; `identification/movement.py` lo hace por el
  historial de alturas ADS-B y sirve de **etiqueta de referencia** para entrenar y medir
  el clasificador de visión. Usar la segunda para rellenar la primera sería evaluar el
  sistema contra sí mismo.
- **La ficha solo lleva datos fijos**: modelo, matrícula, origen, destino. **Nunca**
  altura, velocidad ni posición.
- **Dataset:** división train/val/test **por clips completos**, nunca por fotogramas.
  Los clips **no se publican** (fuera de git).

## Deep learning: aprobado, pero secundario

El profesor **aprobó el uso de deep learning** (20/11/2026), con una condición que manda
sobre todo lo demás: **esto sigue siendo un proyecto de visión por ordenador
tradicional**. En la práctica:

- **los backends por defecto NO cambian**: siguen siendo `classic` (detección) y `bow`
  (clasificación). No los cambies;
- `coco` y `cnn` son **alternativas opcionales** con las que comparar en el informe, no
  el camino principal;
- **no se empiezan hasta que la línea clásica esté terminada y medida**. Sin `bow`
  entrenado no hay con qué comparar la CNN, y la comparación es justamente lo que la
  hace interesante en la memoria;
- los frameworks **no entran en el entorno base**: van en el extra opcional
  `pip install -e ".[dl]"` (ver `pyproject.toml`), para que `environment.yml` siga
  siendo el entorno de la asignatura;
- si implementas `coco`, fija antes `config.detection.coco_label_convention`: hay **dos**
  convenciones de etiquetas de COCO y no coinciden (`airplane` es la 4 en las 80 clases
  contiguas de YOLO y la 5 en los ids originales que usan los modelos de torchvision).
  Equivocarse no da ningún error, simplemente devuelve otra clase.

Sigue en pie: **no añadas `opencv-contrib-python`**. El proyecto usa solo el paquete base
de OpenCV; SIFT está en el módulo principal desde la 4.4.

## Regla general: priorizar las prácticas de la asignatura

Antes de proponer una técnica, mira si hay una equivalente en las prácticas de la
asignatura y úsala. Si una técnica **no** se vio en clase, dilo explícitamente en el
docstring con la nota *"Fuera de las prácticas: justificar con bibliografía en el
informe"*.

Prácticas de referencia: P1 color y HSV, P2 calibración, P3 morfología, P4 contornos,
P5 SIFT/bolsa de palabras/flujo óptico/meanShift, P6 homografías.

## Convenciones de estilo

El estilo es el de las prácticas de la asignatura:

- **identificadores en inglés**, **comentarios y docstrings en español**;
- **type hints en todo**, con `from __future__ import annotations` en cada módulo;
- docstrings estilo Google: `Args:`, `Returns:`, `Raises:`;
- helpers reutilizables (`load_images`, `show_image`, `write_image`) en `common/`;
- **copia la imagen antes de dibujar**: las funciones de dibujo de OpenCV modifican la
  entrada;
- los ficheros fuente se escriben **sin tildes** (los textos de cara al usuario, como
  README y mensajes de los scripts, sí las llevan);
- **ningún valor mágico repartido por el código**: todo parámetro va en `config.py`, en
  la dataclass de su módulo.

Más convenciones:

- los módulos con varios backends tienen una **clase base abstracta** y una **fábrica**
  que elige el backend desde la configuración;
- cada stub lleva type hints, un docstring que explica qué hará, sus entradas y salidas
  y de qué práctica sale la técnica, y un cuerpo `raise NotImplementedError` con un
  mensaje accionable;
- cada pipeline acepta un `DebugRecorder` opcional (`common/debug.py`) y le entrega sus
  imágenes intermedias: de ahí sale la figura de la secuencia de transformación del
  informe;
- las credenciales se leen de `os.environ`, nunca del código;
- el backend de la webcam se resuelve según el sistema operativo (`cv2.CAP_DSHOW` solo
  existe en Windows): está en `CameraConfig.backend`.

## Comandos

```bash
# Entorno
conda env create -f environment.yml
conda activate lanzarote-tracker
pip install -e .

# Comprobaciones
python scripts/check_environment.py
python -m pytest -q
python -m pytest tests/test_config.py -q

# Scripts que ya funcionan
python scripts/camera_test.py
python scripts/generate_cards.py --sheet
python scripts/tune_hsv.py
python scripts/main.py
```

En esta máquina el entorno de la asignatura se llama `voi-lab`
(`C:\Users\javim\anaconda3\envs\voi-lab`), con Python 3.9.21, OpenCV 4.8.0 y NumPy
1.26.0. `cv2.TrackerKCF_create`, `cv2.TrackerCSRT_create` y `cv2.legacy` **no** están
disponibles ahí; sí `cv2.TrackerMIL_create` y `cv2.TrackerGOTURN_create`.

## Git

- **No reescribas el historial.** El enunciado exige que el primer commit contenga la
  lectura de vídeo desde la cámara.
- Commits pequeños, con mensajes del tipo `feat(security): add sequence FSM skeleton`.
- **No hagas push sin preguntar.**
- Si mueves un fichero, usa `git mv` para conservar su historia.

## Tests

`tests/` con pytest, todos reales y sin `xfail` pendientes. Cubren la configuración, la
importación de todos los módulos, la calibración (con un tablero sintético), la
identificación (con dobles, sin red), la captura (con clips sintéticos) y la máquina de
estados del desbloqueo.

Si implementas algo que hoy es un stub, mira antes si ya hay tests escritos contra su
interfaz: varios módulos los tienen.
