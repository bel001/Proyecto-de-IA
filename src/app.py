from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Permitir ejecucion directa (python src/app.py) sin paquete
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import numpy as np

from src.controller import CursorController
from src.gestures import GestureEngine
from src.hand_tracking import HandDetection, HandTracker
from src.smoothing import ExponentialSmoother, MovingAverageSmoother, OneEuroFilter


def _map_to_screen(point: Tuple[float, float], screen_size: Tuple[int, int], gain: float) -> Tuple[float, float]:
    # Aplica ganancia para hacer el movimiento mas amplio sin requerir desplazamientos grandes de la mano.
    x = point[0] * screen_size[0] * gain
    y = point[1] * screen_size[1] * gain
    return x, y


def _draw_overlay(frame: np.ndarray, hand: Optional[HandDetection], status: str) -> None:
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if hand:
        h, w = frame.shape[:2]
        xmin, ymin, xmax, ymax = hand.bbox
        cv2.rectangle(
            frame,
            (int(xmin * w), int(ymin * h)),
            (int(xmax * w), int(ymax * h)),
            (0, 255, 255),
            2,
        )
        idx_tip = hand.landmarks[8]
        cv2.circle(frame, (int(idx_tip[0] * w), int(idx_tip[1] * h)), 8, (255, 0, 0), -1)


def main() -> int:
    parser = argparse.ArgumentParser(description="El Mouse Invisible - control de cursor con gestos.")
    parser.add_argument("--camera-index", type=int, default=0, help="Indice de la camara (por defecto 0).")
    parser.add_argument("--no-flip", action="store_true", help="No espejar la imagen horizontalmente (por defecto si se espeja).")
    parser.add_argument("--alpha", type=float, default=0.35, help="Alpha de suavizado exponencial (0,1]. Mayor = mas rapido.")
    parser.add_argument("--deadzone", type=float, default=0.0015, help="Zona muerta (fraccion de pantalla) para reducir jitter.")
    parser.add_argument("--gain", type=float, default=1.4, help="Ganancia del movimiento para hacerlo mas fluido/amplio.")
    parser.add_argument("--smoothing", choices=["oneeuro", "combo"], default="oneeuro", help="Filtro de suavizado: 'oneeuro' (por defecto) o 'combo' (media movil + exponencial).")
    parser.add_argument("--min-cutoff", type=float, default=1.0, help="Min cutoff del One Euro filter (mas alto = mas suave).")
    parser.add_argument("--beta", type=float, default=0.12, help="Beta del One Euro filter (mas alto = mas reactivo en movimientos rapidos).")
    parser.add_argument("--d-cutoff", type=float, default=1.0, help="Derivative cutoff del One Euro filter.")
    parser.add_argument("--max-step", type=float, default=0.06, help="Paso maximo permitido por frame (fraccion de pantalla) para evitar saltos.")
    parser.add_argument("--max-acc", type=float, default=0.04, help="Cambio maximo de velocidad por frame (fraccion de pantalla) para evitar aceleraciones bruscas.")
    parser.add_argument("--lerp", type=float, default=0.85, help="Factor de interpolacion hacia el punto objetivo (0-1); menor = mas suave, mayor = mas directo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/gesture_classifier.pkl"),
        help="Modelo entrenado opcional (joblib) para gestos.",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("No se pudo abrir la camara.", file=sys.stderr)
        return 1

    tracker = HandTracker()
    gestures = GestureEngine(model_path=args.model)
    controller = CursorController()
    ma_smoother = MovingAverageSmoother(window=5)
    exp_smoother = ExponentialSmoother(alpha=args.alpha)
    one_euro = OneEuroFilter(min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=args.d_cutoff)

    paused = False
    last_point: Optional[Tuple[float, float]] = None
    last_delta: Optional[Tuple[float, float]] = None
    lost_frames = 0
    last_seen_ms = 0.0

    print("Controles: q para salir, p para pausar/reanudar.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame de la camara.", file=sys.stderr)
                break

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            detections = tracker.process(frame)
            hand: Optional[HandDetection] = detections[0] if detections else None
            if hand:
                lost_frames = 0
                last_seen_ms = time.time() * 1000
            else:
                lost_frames += 1

            cmd = gestures.evaluate(hand if not paused else None)

            status_parts = []
            if paused:
                status_parts.append("[PAUSADO]")
            if cmd.gesture_label:
                status_parts.append(f"modelo:{cmd.gesture_label}")
            status = " ".join(status_parts) if status_parts else "EN VIVO"

            if not paused and cmd.move_point:
                target = _map_to_screen(cmd.move_point, (controller.screen_width, controller.screen_height), args.gain)
                if args.smoothing == "oneeuro":
                    smoothed = one_euro.update(target)
                else:
                    averaged = ma_smoother.update(target)
                    smoothed = exp_smoother.update(averaged)
                final_point = smoothed
                if last_point is not None:
                    dx = smoothed[0] - last_point[0]
                    dy = smoothed[1] - last_point[1]
                    norm = math.hypot(dx / controller.screen_width, dy / controller.screen_height)
                    # limitar el paso maximo
                    if norm > args.max_step:
                        scale = (args.max_step / norm) if norm > 1e-6 else 0.0
                        dx *= scale
                        dy *= scale
                    # limitar la aceleracion (cambio de delta)
                    if last_delta is not None:
                        ddx = dx - last_delta[0]
                        ddy = dy - last_delta[1]
                        dnorm = math.hypot(ddx / controller.screen_width, ddy / controller.screen_height)
                        if dnorm > args.max_acc:
                            scale = (args.max_acc / dnorm) if dnorm > 1e-6 else 0.0
                            dx = last_delta[0] + ddx * scale
                            dy = last_delta[1] + ddy * scale
                    final_point = (last_point[0] + dx, last_point[1] + dy)
                    # zona muerta
                    if abs(dx) / controller.screen_width < args.deadzone and abs(dy) / controller.screen_height < args.deadzone:
                        final_point = last_point
                    # interpolacion para suavizar llegada
                    lx, ly = last_point
                    fx, fy = final_point
                    final_point = (lx + (fx - lx) * args.lerp, ly + (fy - ly) * args.lerp)
                controller.move(final_point)
                new_delta = None
                if last_point is not None:
                    new_delta = (final_point[0] - last_point[0], final_point[1] - last_point[1])
                last_point = final_point
                last_delta = new_delta
            elif not paused and hand is None and last_point is not None and lost_frames <= 5:
                # Mantener la posicion reciente si la deteccion se pierde brevemente.
                controller.move(last_point)
            else:
                ma_smoother.reset()
                exp_smoother.reset()
                one_euro.reset()
                last_point = None
                last_delta = None

            if not paused:
                if cmd.drag_start:
                    controller.drag_start()
                if cmd.drag_end:
                    controller.drag_end()
                if cmd.click_left:
                    controller.click_left()
                if cmd.click_right:
                    controller.click_right()
                if cmd.scroll:
                    controller.scroll(cmd.scroll)

            _draw_overlay(frame, hand, status)
            cv2.imshow("El Mouse Invisible", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                if paused:
                    gestures.reset()
                    ma_smoother.reset()
                    exp_smoother.reset()
                    one_euro.reset()
                    last_point = None

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
