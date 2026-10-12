# Inferencia sobre test

- Pesos: `models/yolo26n_b_imgsz640_best.pt`
- Imágenes: 31 (`dataset/images/test`)
- `imgsz` = 640, confianza mínima = 0.25, IoU para considerar acierto = 0.5

| Resultado por imagen | Imágenes |
|---|---|
| Correcta: todas las placas detectadas, sin falsos positivos | 18 |
| Con falsos positivos: detecta algo que no es una placa | 8 |
| Con placas no detectadas (falsos negativos) | 0 |
| Con falsos positivos y placas no detectadas | 2 |
| Sin detecciones | 3 |

| Cajas | |
|---|---|
| Verdaderos positivos | 27 |
| Falsos positivos | 13 |
| Placas no detectadas | 5 |
| Precision | 0.675 |
| Recall | 0.844 |

Colores en la galería: verde = acierto, rojo = falso positivo, amarillo = placa real no detectada (dibujada desde la etiqueta).

- **correct**: Cars100.png, Cars113.png, Cars117.png, Cars118.png, Cars15.png, Cars165.png, Cars18.png, Cars192.png, Cars226.png, Cars243.png, Cars309.png, Cars373.png, Cars375.png, Cars394.png, Cars51.png, Cars79.png, Cars88.png, Cars9.png
- **false_positive**: Cars152.png, Cars16.png, Cars19.png, Cars227.png, Cars228.png, Cars261.png, Cars263.png, Cars311.png
- **mixed**: Cars215.png, Cars236.png
- **no_detections**: Cars114.png, Cars23.png, Cars75.png
