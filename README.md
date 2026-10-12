# YOLO License Plates

Detección de matrículas con YOLO entrenado **desde pesos aleatorios** en CPU.

## Modelo

| | |
|---|---|
| Librería | Ultralytics **8.4.175** (fijada en `poetry.lock`) |
| Arquitectura | **YOLO26n** (`yolo26n.yaml`): la variante más pequeña de YOLO26, la generación más reciente de esta versión estable. Unos 2.5 M de parámetros y 5.9 GFLOPs, adecuada para CPU. |
| Inicialización | Pesos aleatorios, sin pesos preentrenados |
| Hardware | CPU de un Apple M5 (4 núcleos de rendimiento + 6 de eficiencia, 16 GB), sin GPU. PyTorch `2.14.1`. |

### Arquitectura vs. pesos entrenados

Ultralytics acepta dos tipos de archivo en `YOLO(...)`:

| | `yolo26n.yaml` | `yolo26n.pt` |
|---|---|---|
| Qué es | Definición de la arquitectura: capas, conexiones y escalas | Checkpoint: arquitectura + pesos entrenados + metadatos |
| Pesos al cargar | Aleatorios | Preentrenados (COCO) |
| `model.ckpt` | `{}` (vacío) | Dict con `epoch`, `train_args`, `date`, `version`… |
| `model.ckpt_path` | `None` | Ruta al `.pt` |

El programa de entrenamiento lo garantiza de tres formas:

1. `load_config` rechaza cualquier `model` que no termine en `.yaml` o `.yml`.
2. Antes de entrenar comprueba que `model.ckpt` esté vacío y que `ckpt_path` sea `None`. Si no, aborta.
3. Llama a `train(..., pretrained=False)`, lo que garantiza que el trainer no cargue pesos (`engine/trainer.py`, `setup_model`).

El resultado de esta comprobación queda guardado en `run_info.json` → `weights_check`.

## Datos

`dataset/` parte de las 433 imágenes de [dataset-matriculas] (clase única `licence`).

**Duplicados.** El dataset original tiene **131 copias byte a byte** de otras imágenes: solo 302 fotos son distintas, y algunas aparecen hasta 5 veces con otro nombre. En una primera división sin deduplicar, 23 de las 43 imágenes de test tenían una copia exacta en train o val, y 37 de las 87 de val una en train, así que las métricas medían memorización. Por ejemplo, en test el recall era 0.957 en las imágenes con copia y 0.750 en las nunca vistas.

Por eso `split_dataset` elimina los duplicados antes de dividir. De cada grupo se queda la copia con el número más bajo, con su propia anotación, y las descartadas quedan listadas en `dataset/duplicates.csv`. Las copias no tenían anotaciones idénticas: el IoU mediano entre sus cajas es 0.83 y en 2 grupos ni siquiera coincide el número de cajas. Ese ruido de anotación también pone un techo a métricas estrictas como mAP50-95.

- **División:** 302 imágenes únicas con 334 matrículas, 70/20/10 con semilla 42.
  - train: 211 imágenes (233 cajas)
  - val: 60 (69)
  - test: 31 (32)
- **Estructura:**
  - `images/{split}/`: imágenes.
  - `annotations/{split}/`: XML Pascal VOC originales.
  - `labels/{split}/`: cajas YOLO.
  - `data.yaml`: configuración del dataset.
  - `duplicates.csv`: copias descartadas.

Para regenerarlo:

```bash
poetry run python -m yolo_license_plates.data.split_dataset \
  ~/Downloads/dataset-matriculas/images --annotations ~/Downloads/dataset-matriculas/annotations
poetry run python -m yolo_license_plates.data.voc_to_yolo
```

## Entrenamiento

Cada experimento es un YAML en `configs/experiments/`. Para repetirlo en local:

```bash
poetry install
poetry run python -m yolo_license_plates.training.train --config configs/experiments/smoke.yaml       # comprobación rápida
poetry run python -m yolo_license_plates.training.train --config configs/experiments/a_imgsz416.yaml
poetry run python -m yolo_license_plates.training.train --config configs/experiments/b_imgsz640.yaml
```

Las flags de la línea de comandos sobrescriben la config, por ejemplo `--epochs 5 --name prueba`.

También se puede correr en Docker (`docker compose run --rm trainer --config ...`), pero en este Mac es más lento. En la misma prueba `smoke`, Docker tardó ~26.6 s por época con 4 CPUs y el Mac ~16.5 s. Darle 8 CPUs a la VM no acelera: PyTorch reparte el trabajo en partes iguales entre hilos y los núcleos de eficiencia frenan a los de rendimiento.

- **Semilla:** fija (`seed: 42`, `deterministic: true`) y sin `cache` en RAM. Dos corridas de `smoke` dieron pérdidas y métricas idénticas en cada época.
- **Límite de la reproducibilidad:** el resultado exacto depende del número de hilos y de la plataforma. Por ejemplo, la misma config en Docker con 4 y con 8 CPUs dio métricas distintas desde la primera época. Para repetir un experimento hay que usar el mismo entorno que registra `run_info.json` → `hardware` y `software`.
- **Salida:** cada corrida escribe en `runs/<name>/`:
  - `weights/best.pt` y `weights/last.pt`.
  - `results.csv` y `results.png`: pérdidas y métricas por época.
  - Curvas PR, P, R y F1, y matriz de confusión.
  - `run_info.json`: épocas pedidas y completadas, mejor época, `imgsz`, `batch`, tiempo total y por época, hardware, versiones y métricas de val del mejor modelo.

### Comparación

Las configuraciones A y B solo difieren en **`imgsz`** (416 vs 640); un test lo comprueba. La elección se hace con el **mAP50-95 en val**. El split test no se usa para elegir.

```bash
poetry run python -m yolo_license_plates.training.compare runs/a_imgsz416 runs/b_imgsz640
```

El script genera `reports/comparacion.md`.

## Evaluación sobre test

Con los pesos elegidos se hace inferencia sobre las 31 imágenes de test:

```bash
poetry run python -m yolo_license_plates.evaluation.predict_gallery --weights models/<elegido>.pt
```

El `imgsz` de inferencia es, por defecto, el mismo con el que se entrenaron los pesos. El script escribe en `reports/test_predictions/`:

- `predictions/<imagen>.txt`: predicciones en formato YOLO con confianza (`clase xc yc w h conf`).
- `predictions.json`: las mismas cajas en píxeles, cada una marcada como acierto o falso positivo, y las placas no detectadas.
- `gallery/<resultado>/<imagen>.jpg`: la imagen original sin marcas a la izquierda y la predicción con cajas y confianza a la derecha.
- `index.html`: galería navegable agrupada por resultado.
- `summary.md`: conteos y precision/recall con `conf` 0.25 e IoU 0.5.

**Cómo se clasifica cada imagen.** Cada predicción se compara con las etiquetas reales: es un acierto si su IoU con una placa es ≥ 0.5. El emparejamiento recorre las predicciones de mayor a menor confianza, y cada placa solo puede emparejarse con una. Así, una detección duplicada de la misma placa cuenta como falso positivo. Según el resultado, la imagen va a una de estas carpetas de la galería:

| Carpeta | Significado |
|---|---|
| `correct` | Todas las placas detectadas, sin falsos positivos |
| `false_positive` | Hay detecciones que no corresponden a ninguna placa (incluidos duplicados) |
| `missed` | Alguna placa sin detectar |
| `mixed` | Falsos positivos y placas sin detectar |
| `no_detections` | El modelo no detectó nada |

**Colores en la imagen de la derecha:**
- verde: acierto;
- rojo: falso positivo, con la etiqueta debajo de la caja;
- amarillo: placa real no detectada, dibujada desde la etiqueta para que el error se vea.

[dataset-matriculas]: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
