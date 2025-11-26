from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.hand_tracking import HandTracker  # noqa: E402
from src.gestures import normalize_landmarks  # noqa: E402


def _columns() -> List[str]:
    cols = []
    for i in range(21):
        cols.extend([f"l{i}_x", f"l{i}_y", f"l{i}_z"])
    cols.append("label")
    return cols


def main() -> int:
    parser = argparse.ArgumentParser(description="Recolecta landmarks etiquetados para gestos.")
    parser.add_argument("--label", required=True, help="Etiqueta a capturar (ej. click_left).")
    parser.add_argument("--frames", type=int, default=200, help="Numero de frames a guardar.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--flip", action="store_true", help="Espejar imagen horizontalmente.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Carpeta de salida.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("No se pudo abrir la camara.", file=sys.stderr)
        return 1

    tracker = HandTracker()
    rows: List[List[float]] = []
    columns = _columns()
    target = args.frames
    print(f"Grabando {target} frames para la etiqueta '{args.label}'. Presiona q para cancelar.")

    try:
        while len(rows) < target:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame.", file=sys.stderr)
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            detections = tracker.process(frame)
            if detections:
                norm = normalize_landmarks(detections[0]).flatten().tolist()
                norm.append(args.label)
                rows.append(norm)

            progress = f"{len(rows)}/{target} frames"
            cv2.putText(frame, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Recolectar datos", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Cancelado por el usuario.")
                break

        if rows:
            timestamp = int(time.time())
            out_path = args.output_dir / f"{args.label}_{timestamp}.csv"
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(out_path, index=False)
            print(f"Guardado: {out_path}")
        else:
            print("No se recolectaron datos.")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
