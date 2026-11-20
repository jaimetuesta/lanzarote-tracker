# `data/` — clips, fotogramas, anotaciones y splits

**Este directorio está fuera de git**, salvo este README y los `.gitkeep`. Los clips
grabados del directo del aeropuerto **no se publican**.

## Estructura esperada

```
data/
├── clips/        clips grabados del directo, con su JSON de metadatos
├── frames/       fotogramas extraídos, un subdirectorio por clip
├── annotations/  etiquetas de orientación (frente / espaldas)
└── splits/       división train / val / test, por clips completos
```

## `clips/`

Cada clip se guarda con `cv2.VideoWriter` y va acompañado de un JSON con el mismo
nombre:

```
clips/
├── clip_20261120T173000Z.mp4
├── clip_20261120T173000Z.json
├── clip_20261120T174500Z.mp4
└── clip_20261120T174500Z.json
```

El nombre lleva la marca de tiempo **UTC** de inicio en formato
`%Y%m%dT%H%M%SZ`. El JSON contiene:

```json
{
  "clip_path": "data/clips/clip_20261120T173000Z.mp4",
  "started_at_utc": "2026-11-20T17:30:00Z",
  "duration_seconds": 60.0,
  "fps": 25.0,
  "width": 1920,
  "height": 1080,
  "frame_count": 1500,
  "source_url": "https://www.youtube.com/watch?v=..."
}
```

La marca de tiempo UTC es imprescindible: es lo que permite cruzar el clip con las
consultas de OpenSky, que trabajan con marcas de tiempo UNIX.

Los genera `python scripts/record_clips.py`.

## `frames/`

Un subdirectorio por clip, con el mismo nombre que el clip. **Nunca se mezclan los
fotogramas de varios clips en un mismo directorio**: saber de qué clip viene cada
fotograma es lo que permite dividir el dataset correctamente.

```
frames/
├── clip_20261120T173000Z/
│   ├── frame_00000.png
│   ├── frame_00005.png
│   └── ...
└── clip_20261120T174500Z/
    └── ...
```

Los genera `python scripts/extract_frames.py`, que guarda uno de cada
`CaptureConfig.frame_step` fotogramas.

## `annotations/`

Etiquetas de orientación de cada fotograma o recorte: `front` (el avión viene hacia la
cámara) o `back` (se aleja). Un fichero por clip, con el mismo nombre que el clip.

## `splits/`

Un fichero por partición, con **un nombre de clip por línea**:

```
splits/
├── train.txt
├── val.txt
└── test.txt
```

## Regla: la división es por clips completos

**La división train/val/test se hace siempre por clips completos, nunca por
fotogramas.**

Dos fotogramas consecutivos del mismo clip son casi idénticos: mismo avión, misma luz,
mismo encuadre, a unas centésimas de distancia. Si se repartieran entre particiones, el
clasificador vería en entrenamiento imágenes prácticamente iguales a las de validación,
y las métricas saldrían infladas sin que el modelo haya generalizado nada.

Por eso `scripts/build_dataset.py` baraja **nombres de clip**, no fotogramas, y escribe
las particiones como listas de clips.
