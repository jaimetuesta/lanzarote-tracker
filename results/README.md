# `results/` — salidas de cada módulo

Un subdirectorio por módulo, con las salidas que genera al ejecutarlo:

```
results/
├── calibration/   figuras antes/después de la corrección de distorsión
├── security/      secuencia de transformación del pipeline de las tarjetas
├── detection/     máscaras y contornos del detector del avión
├── tracker/       vídeo anotado y trayectorias del seguimiento
└── classification/ puntos SIFT y matrices de confusión
```

## Secuencia de transformación

El informe exige una figura con **todas** las transformaciones que sufre la imagen. La
genera el `DebugRecorder` de `common/debug.py`: al ejecutar un script con `--debug`,
cada pipeline entrega sus imágenes intermedias y se guardan aquí numeradas en orden,
más un mosaico con todos los pasos en una sola figura.

```bash
python scripts/run_security.py --debug     # -> results/security/00_original.png, ...
python scripts/run_tracker.py CLIP --debug # -> results/tracker/...
```

## Qué se versiona

La estructura de directorios sí se versiona. Las imágenes que se usen en el informe
conviene conservarlas; los vídeos quedan fuera por la regla `*.mp4` del `.gitignore`.
