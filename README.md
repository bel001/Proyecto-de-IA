# El Mouse Invisible

Control de cursor y gestos con una webcam utilizando Python, OpenCV, MediaPipe y acciones de sistema operativo con `pyautogui`. Diseñado para accesibilidad y para contextos sin contacto fisico.

## Caracteristicas
- Seguimiento de mano en tiempo real con MediaPipe Hands.
- Mapeo de dedo indice al cursor, con suavizado y zona muerta.
- Gestos sin contacto:
  - Pinza pulgar+indice: clic izquierdo (o arrastre si se mantiene).
  - Pinza pulgar+medio: clic derecho.
  - Doble dedo (indice+medio) extendido: modo scroll; desplazar la mano arriba/abajo genera `scroll`.
  - Gesto de pausa (tecla `p` o sin mano en pantalla) para ceder control.
- Scripts para recoleccion de datos de gestos y entrenamiento de un clasificador (RandomForest) con landmarks.

## Requisitos
- Python 3.10 o 3.11 (recomendado; MediaPipe no publica wheel para 3.13 en Windows aun).
- Webcam funcional.
- Dependencias: ver `requirements.txt` (mediapipe 0.10.14, opencv-python 4.12.0.88, etc.).

Instalacion rapida (Windows):
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar el demo en tiempo real
```powershell
python src/app.py --smoothing oneeuro --min-cutoff 1.0 --beta 0.12 --deadzone 0.0015 --gain 1.4 --max-step 0.06 --max-acc 0.04 --lerp 0.85
```
Opciones utiles:
- `--gain`: multiplica el desplazamiento para que el cursor se mueva mas con menos recorrido de la mano (por defecto 1.2).
- `--alpha`: suavizado (0,1]; mayor = mas reactivo, menor = mas suave.
- `--deadzone`: reduce jitter; si el cursor se siente “pegado”, baja a 0.002–0.001. Para mas estabilidad sube a 0.003–0.005.
- `--max-step`: limita el salto maximo por frame (fraccion de pantalla) para evitar teletransportes; bajar a 0.04 si aun hay saltos grandes.
- `--max-acc`: limita el cambio de velocidad por frame (fraccion de pantalla) para evitar acelerones.
- `--lerp`: interpola hacia el destino; mayor valor = mas directo/rapido, menor = mas suave/lento.
- Vista: por defecto la camara se espeja (modo espejo). Usa `--no-flip` si no quieres invertir la vista.
- Suavizado:
  - `--smoothing oneeuro` (default): filtro One Euro para respuesta rapida con poco jitter. Ajusta `--min-cutoff` (mas alto = mas suave), `--beta` (mas alto = mas reactivo en movimientos rapidos) y `--d-cutoff`.
  - `--smoothing combo`: media movil + exponencial (alpha) como antes.

Atajos:
- `q`: salir.
- `p`: pausar/reanudar control.

## Scripts de datos y entrenamiento
- Recolectar landmarks etiquetados:
  ```powershell
  python tools/collect_data.py --label click_left --frames 150
  ```
  Los archivos se guardan en `data/raw/` como CSV.

- Entrenar clasificador de gestos (RandomForest) con los CSV de `data/raw/`:
  ```powershell
  python tools/train_classifier.py --data-dir data/raw --output models/gesture_classifier.pkl --cv-folds 5
  ```
  El modelo se carga automaticamente en `app.py` si existe (mejora la deteccion frente a las reglas por defecto).

## Evaluacion de desempeño del sistema (FPS/latencia)
- Ejecuta la medicion sin mover el cursor, solo midiendo el pipeline:
  ```powershell
  python tools/evaluate_system.py --frames 300 --flip --save-csv metrics.csv
  ```
  Reporta FPS, latencias promedio/p95 y la tasa de frames con mano detectada; opcionalmente guarda las metricas por frame en CSV.

## Estructura
- `src/app.py`: bucle principal de captura y control del cursor.
- `src/hand_tracking.py`: envoltorio de MediaPipe Hands.
- `src/gestures.py`: heuristicas de gestos y compatibilidad con modelo entrenado.
- `src/controller.py`: acciones de cursor/clic/scroll via `pyautogui`.
- `src/smoothing.py`: suavizado del movimiento.
- `tools/collect_data.py`: captura de landmarks etiquetados.
- `tools/train_classifier.py`: entrenamiento clasico sobre features geometricos.
- `data/`: carpeta para datasets (se crea con `.gitkeep`).
- `models/`: modelos entrenados (se crea con `.gitkeep`).

## Notas de uso
- La iluminacion uniforme mejora la estabilidad.
- Ajusta los umbrales en `src/gestures.py` y la sensibilidad/alpha en `src/app.py` segun tu camara.
- Usa la tecla `p` para pausar el control antes de hacer clic con el mouse fisico.
