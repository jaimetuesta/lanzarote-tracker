# `report/` — informe del proyecto

Aquí va la **plantilla LaTeX de la asignatura**, que proporcionará Javier. Hasta
entonces este directorio solo contiene este README.

**Máximo 5 páginas.**

## Estructura del informe

1. **Introducción**
   - Qué problema se aborda y por qué.
   - La cámara de Flightradar24 del aeropuerto de Lanzarote: gira y hace zoom, lo que
     condiciona todas las decisiones técnicas.
   - Alcance base frente a ampliaciones.

2. **Metodología**
   1. **Calibración** — tablero, modelo pinhole, K, coeficientes de distorsión y RMS de
      reproyección. Tabla con los valores obtenidos y figura antes/después de la
      corrección.
   2. **Diagrama de bloques** — el del README, con el sistema de seguridad y el sistema
      propuesto, y las ampliaciones aparte.
   3. **Secuencia de transformación** — figura obligatoria con **todas** las
      transformaciones que sufre la imagen. Se genera con el `DebugRecorder`
      (`common/debug.py`): al ejecutar los scripts con `--debug` guarda cada paso
      numerado en `results/<módulo>/` y además un mosaico.
   4. **Sistema de seguridad** — segmentación HSV (con los dos rangos del rojo),
      morfología, contorno de mayor área, forma por `approxPolyDP` y circularidad,
      validación temporal de N fotogramas y máquina de estados del desbloqueo.
   5. **Sistema propuesto** — captura de clips, detección del avión, seguimiento
      detectar-cada-K + flujo óptico, clasificación frente/espaldas, deducción de
      llegada o salida e identificación con OpenSky.
   6. **Ampliaciones** — qué harían y por qué quedan fuera del alcance base.
   7. **Salida de vídeo** — recuadro, ficha de datos fijos y FPS incrustado.

3. **Resultados**
   - Métricas de la clasificación sobre la partición de test (dividida **por clips
     completos**).
   - Comparación de backends: `classic` frente a `coco` y `bow` frente a `cnn`, si se
     llegan a aprobar.
   - Comparación de seguidores: flujo óptico Lucas-Kanade frente a meanShift, con el
     zoom de la cámara como caso difícil.
   - FPS medio alcanzado.

4. **Futuros desarrollos**
   - Las cuatro ampliaciones, por orden de prioridad.
   - Limitaciones conocidas: las consultas de llegadas/salidas de OpenSky solo están
     disponibles al día siguiente; varios aviones simultáneos en el recuadro geográfico.

## Qué hay que justificar con bibliografía

Toda técnica que **no** se haya visto en las prácticas de la asignatura. Los stubs del
código la señalan con la nota *"Fuera de las prácticas: justificar con bibliografía en
el informe"*. A día de hoy: el filtro de Kalman, la estabilización por homografía, los
descriptores de textura y el criterio que relaciona orientación y cabecera en uso.

## Figuras

Las figuras salen de `results/` y de `notebooks/`. Las de la secuencia de transformación
se generan ejecutando los scripts con `--debug`.
