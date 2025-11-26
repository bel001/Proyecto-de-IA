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

from src.gestures import GestureEngine  # noqa: E402
from src.hand_tracking import HandDetection, HandTracker  # noqa: E402
from src.smoothing import OneEuroFilter  # noqa: E402


def _normalize_move_point(hand: HandDetection) -> np.ndarray:
    # Devuelve la punta del indice en coords normalizadas absolutas.
    x, y, _ = hand.landmarks[8]
    return np.array([x, y], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalua desempeño (FPS/latencia) del pipeline sin mover el cursor.")
    parser.add_argument("--camera-index", type=int, default=0, help="Indice de la camara (por defecto 0).")
    parser.add_argument("--frames", type=int, default=300, help="Numero de frames a medir.")
    parser.add_argument("--flip", action="store_true", help="Espejar la imagen horizontalmente.")
    parser.add_argument("--model", type=Path, default=Path("models/gesture_classifier.pkl"), help="Modelo opcional.")
    parser.add_argument("--save-csv", type=Path, help="Ruta para guardar las metricas por frame (CSV).")
    parser.add_argument("--min-cutoff", type=float, default=1.0, help="Min cutoff One Euro (suavizado).")
    parser.add_argument("--beta", type=float, default=0.12, help="Beta One Euro (respuesta).")
    parser.add_argument("--d-cutoff", type=float, default=1.0, help="Derivative cutoff One Euro.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("No se pudo abrir la camara.", file=sys.stderr)
        return 1

    tracker = HandTracker()
    gestures = GestureEngine(model_path=args.model)
    smoother = OneEuroFilter(min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=args.d_cutoff)

    per_frame: List[dict] = []
    start_global = time.perf_counter()

    try:
        for i in range(args.frames):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame de la camara.", file=sys.stderr)
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            t1 = time.perf_counter()
            detections = tracker.process(frame)
            hand = detections[0] if detections else None
            t2 = time.perf_counter()

            cmd_time = 0.0
            smooth_time = 0.0
            has_hand = 0
            if hand:
                has_hand = 1
                t_cmd0 = time.perf_counter()
                cmd = gestures.evaluate(hand)
                t_cmd1 = time.perf_counter()
                cmd_time = (t_cmd1 - t_cmd0) * 1000
                if cmd.move_point:
                    t_sm0 = time.perf_counter()
                    smoother.update(cmd.move_point)
                    t_sm1 = time.perf_counter()
                    smooth_time = (t_sm1 - t_sm0) * 1000

            t3 = time.perf_counter()
            per_frame.append(
                {
                    "frame": i,
                    "capture_ms": (t1 - t0) * 1000,
                    "detect_ms": (t2 - t1) * 1000,
                    "command_ms": cmd_time,
                    "smooth_ms": smooth_time,
                    "total_ms": (t3 - t0) * 1000,
                    "has_hand": has_hand,
                }
            )
    finally:
        tracker.close()
        cap.release()

    total_time = time.perf_counter() - start_global
    n = len(per_frame)
    if n == 0:
        print("No se registraron frames.", file=sys.stderr)
        return 1

    df = pd.DataFrame(per_frame)
    fps = n / total_time
    summary = {
        "frames": n,
        "fps_prom": fps,
        "total_ms_prom": df["total_ms"].mean(),
        "total_ms_p95": df["total_ms"].quantile(0.95),
        "detect_ms_prom": df["detect_ms"].mean(),
        "detect_ms_p95": df["detect_ms"].quantile(0.95),
        "command_ms_prom": df["command_ms"].mean(),
        "has_hand_ratio": df["has_hand"].mean(),
    }

    print("=== Desempeño del sistema ===")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_csv, index=False)
        print(f"Metricas por frame guardadas en {args.save_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
