# Comparación de configuraciones (validación)

Parámetro que cambia: **imgsz**. Criterio: mAP50-95 de `best.pt` en el split **val**.

| | a_imgsz416 | b_imgsz640 |
|---|---|---|
| `model` | yolo26n.yaml | yolo26n.yaml |
| `imgsz` | 416 | 640 |
| `batch` | 16 | 16 |
| `epochs` | 100 | 100 |
| `patience` | 30 | 30 |
| `seed` | 42 | 42 |
| `device` | cpu | cpu |
| Épocas completadas | 100 | 100 |
| Mejor época | 89 | 93 |
| Parada temprana | no | no |
| Tiempo total (min) | 22.9 | 51.7 |
| Segundos por época | 13.7 | 31.0 |
| **Precision** | 0.7527 | 0.8435 |
| **Recall** | 0.7536 | 0.6812 |
| **mAP50** | 0.7581 | 0.8054 |
| **mAP50-95** | 0.348 | 0.3906 |

**Configuración elegida: `b_imgsz640`** (mAP50-95 val = 0.3906).

El tiempo se mide con `time.perf_counter()`, que en macOS no cuenta el tiempo en reposo del equipo. La columna `time` de `results.csv` es tiempo de reloj y sí lo incluye.

## Hardware y software

- `a_imgsz416`: Apple M5, 10 CPUs (4 de rendimiento + 6 de eficiencia), 8 hilos de torch, 17.2 GB RAM, Docker=False; Python 3.11.14, torch 2.14.1, Ultralytics 8.4.175
- `b_imgsz640`: Apple M5, 10 CPUs (4 de rendimiento + 6 de eficiencia), 8 hilos de torch, 17.2 GB RAM, Docker=False; Python 3.11.14, torch 2.14.1, Ultralytics 8.4.175
