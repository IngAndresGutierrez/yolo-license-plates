# Resultados

Modelo YOLO26n (`yolo26n.yaml`) entrenado desde pesos aleatorios en la CPU de un Apple M5, con Ultralytics 8.4.175. Dataset deduplicado: 302 imágenes únicas, divididas en 211 de train, 60 de val y 31 de test.

## Elección de configuración (validación)

Las dos configuraciones solo difieren en `imgsz`. La elección se hace con el mAP50-95 de `best.pt` en val. El detalle completo está en [`comparacion.md`](comparacion.md).

| Val | A · `imgsz` 416 | B · `imgsz` 640 |
|---|---|---|
| Precision | 0.753 | **0.844** |
| Recall | **0.754** | 0.681 |
| mAP50 | 0.758 | **0.805** |
| mAP50-95 | 0.348 | **0.391** |
| Mejor época / completadas | 89 / 100 | 93 / 100 |
| Tiempo de entrenamiento | 22.9 min (13.7 s/época) | 51.7 min (31.0 s/época) |

**Elegida: B (`imgsz` 640).**
- **Por qué:** mejora el mAP50-95 en 4.3 puntos y el mAP50 en 4.7. El 31% de las placas ocupa menos del 1% de la imagen, así que la resolución importa.
- **Costo:** cada época tarda 2.3 veces más, y el recall baja un poco (0.754 → 0.681).

Ninguna configuración se detuvo antes por la paciencia (30 épocas). Las dos alcanzaron su mejor resultado cerca del final (épocas 89 y 93), cuando el learning rate ya es bajo. En B, la mejor época cae dentro de las últimas 10, en las que el mosaic está desactivado (`close_mosaic=10`).

Los pesos elegidos están en [`../models/yolo26n_b_imgsz640_best.pt`](../models/yolo26n_b_imgsz640_best.pt).

## Inferencia sobre test

31 imágenes y 32 placas, con confianza ≥ 0.25 e IoU ≥ 0.5. La galería y las predicciones están en [`test_predictions/`](test_predictions/); `index.html` permite recorrerla.

| Resultado por imagen | Imágenes |
|---|---|
| Correcta | 18 |
| Con falsos positivos | 8 |
| Con placas no detectadas | 0 |
| Mixta | 2 |
| Sin detecciones | 3 |

A nivel de cajas hay 27 aciertos, 13 falsos positivos y 5 placas no detectadas: **precision 0.675, recall 0.844**. Con solo 32 placas, cada error mueve la precision o el recall unos 3 puntos.

## Análisis de errores

- **7 de los 13 falsos positivos son detecciones duplicadas** de una placa que el modelo ya acertó: una segunda caja desplazada sobre la misma placa. YOLO26 no usa NMS en inferencia y, con tan pocos datos, todavía genera duplicados con confianza parecida.
- **Otros falsos positivos son texto que parece una placa,** como el letrero "SUZUKI WESTERN MOTORS" del vidrio trasero en `Cars19`, detectado con confianza 0.55.
- **Algunos "falsos positivos" son errores de anotación.** `Cars215` es un collage con 5 placas, pero la etiqueta solo marca una; las dos detecciones "falsas" son placas reales.
- **Las 3 imágenes sin detecciones fallan por razones distintas:**
  - `Cars75` es una escena de tráfico donde la placa ocupa el 0.05% de la imagen. En train, solo 3 de las 233 placas ocupan menos del 0.1%.
  - `Cars114` es lo contrario: un primer plano de una placa decorativa iluminada que ocupa el 78% de la imagen. Ninguna placa de train ocupa más del 50%.
  - `Cars23` es una placa normal y bien visible en un coche de policía, rodeada de otros textos grandes ("POLICE", "9214", "$10,000 REWARD"). El modelo no la detecta con confianza ≥ 0.25.
- **El mAP50-95 tiene un techo bajo.** Las copias duplicadas del dataset original traían anotaciones distintas de la misma placa, con un IoU mediano de 0.83 entre ellas. Por eso es difícil acertar con los umbrales de IoU 0.90–0.95, aunque la detección sea buena.

## Sobre los datos

El dataset original tenía 131 imágenes duplicadas byte a byte. Sin deduplicar, la misma comparación daba valores más altos (B: mAP50-95 0.469, mAP50 0.896) porque val y test compartían fotos con train. Las métricas de este reporte usan el dataset deduplicado. La lista de copias descartadas está en `dataset/duplicates.csv`.
