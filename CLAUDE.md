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
  preentrenado, clase `airplane`) es opcional y está **desactivado**.
- **Clasificación frente/espaldas:** dos backends tras una interfaz. `bow` (bolsa de
  palabras SIFT + K-means + SVM, Práctica 5 apartado A) es la **línea base**; `cnn` (CNN
  propia) es opcional y está **desactivada**.
- **Captura:** URL del stream con yt-dlp y lectura con `cv2.VideoCapture`; los clips se
  guardan con `cv2.VideoWriter` y un JSON de metadatos con la marca de tiempo UTC de
  inicio.
- **OpenSky:** vectores de estado en un recuadro geográfico alrededor del aeropuerto;
  modelo y matrícula desde su base de datos de aeronaves; origen, destino y horas desde
  las consultas de llegadas/salidas por aeropuerto, que **solo están disponibles al día
  siguiente**. **No** implementes autenticación ni asumas límites de uso: hay un TODO
  explícito para verificarlo en la documentación actual de OpenSky. Las credenciales se
  leen de variables de entorno.
- **La ficha solo lleva datos fijos**: modelo, matrícula, origen, destino. **Nunca**
  altura, velocidad ni posición.
- **Dataset:** división train/val/test **por clips completos**, nunca por fotogramas.
  Los clips **no se publican** (fuera de git).

## Restricciones de deep learning

Los backends `coco` y `cnn` usan deep learning y su uso está **pendiente de aprobación
del profesor**. Mientras no se apruebe:

- su constructor lanza `NotImplementedError` con un mensaje que lo explica;
- **no añadas ningún framework de deep learning** a `environment.yml` ni a
  `requirements.txt`: están comentados con una nota;
- los backends por defecto siguen siendo `classic` y `bow`.

**Tampoco añadas `opencv-contrib-python`.** El proyecto usa solo el paquete base de
OpenCV; SIFT está en el módulo principal desde la 4.4.

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

`tests/` con pytest. De momento hay tests reales de la configuración y de la importación
de todos los módulos. Los de la máquina de estados están escritos contra su interfaz y
marcados como `xfail`: al implementarla pasarán a XPASS y hay que **quitar el
`pytestmark`** de `tests/test_sequence_fsm.py`.
