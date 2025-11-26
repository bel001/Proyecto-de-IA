from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np

from .hand_tracking import HandDetection


PINCH_THRESHOLD = 0.08  # distancia normalizada para detectar pinza (mas tolerante)
DRAG_HOLD_MS = 400
SCROLL_GAIN = 800  # factor para convertir delta vertical a scroll
SCROLL_DEADZONE = 0.005
RIGHT_CLICK_DEBOUNCE_MS = 350
LEFT_CLICK_DEBOUNCE_MS = 250


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def normalize_landmarks(hand: HandDetection) -> np.ndarray:
    """
    Normaliza landmarks al cuadro delimitador de la mano para invarianza a escala/traslacion.
    Retorna array shape (21, 3).
    """
    xmin, ymin, xmax, ymax = hand.bbox
    width = max(xmax - xmin, 1e-3)
    height = max(ymax - ymin, 1e-3)
    norm = []
    for x, y, z in hand.landmarks:
        norm_x = (x - xmin) / width
        norm_y = (y - ymin) / height
        norm.append((norm_x, norm_y, z))
    return np.asarray(norm, dtype=np.float32)


def build_feature_vector(norm_landmarks: np.ndarray) -> np.ndarray:
    """
    Flatten de coords normalizadas + distancias clave para usar en modelos clasicos.
    """
    if norm_landmarks.shape != (21, 3):
        raise ValueError("Se esperaban 21 landmarks con 3 coords.")

    feats = norm_landmarks.flatten().tolist()
    # Distancias de referencia
    pairs = [(4, 8), (4, 12), (4, 16), (8, 12), (8, 16), (12, 16)]
    for a, b in pairs:
        feats.append(_distance(norm_landmarks[a], norm_landmarks[b]))
    return np.asarray(feats, dtype=np.float32)


def _finger_is_up(norm_landmarks: np.ndarray, tip: int, pip: int) -> bool:
    # Coordenada y mas pequena significa mas arriba en la imagen.
    return norm_landmarks[tip][1] < norm_landmarks[pip][1] - 0.01


@dataclass
class GestureCommand:
    move_point: Optional[Tuple[float, float]] = None
    click_left: bool = False
    click_right: bool = False
    drag_start: bool = False
    drag_end: bool = False
    scroll: int = 0
    gesture_label: Optional[str] = None


class GestureEngine:
    """
    Detecta gestos por reglas y opcionalmente por modelo entrenado.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model = None
        if model_path and model_path.exists():
            self.model = joblib.load(model_path)
        self._left_pinch_active = False
        self._left_pinch_start = 0.0
        self._drag_active = False
        self._right_pinch_active = False
        self._last_right_click_ms = 0.0
        self._last_left_click_ms = 0.0
        self._scroll_prev_y: Optional[float] = None

    def reset(self) -> None:
        self._left_pinch_active = False
        self._right_pinch_active = False
        self._drag_active = False
        self._last_right_click_ms = 0.0
        self._last_left_click_ms = 0.0
        self._scroll_prev_y = None

    def _classify_with_model(self, norm_landmarks: np.ndarray) -> Optional[str]:
        if self.model is None:
            return None
        feats = build_feature_vector(norm_landmarks).reshape(1, -1)
        try:
            return str(self.model.predict(feats)[0])
        except Exception:
            return None

    def evaluate(self, hand: Optional[HandDetection]) -> GestureCommand:
        if hand is None:
            self.reset()
            return GestureCommand()

        norm = normalize_landmarks(hand)
        # Para el cursor usamos coordenadas absolutas normalizadas de la imagen (menos jitter que por bbox).
        idx_x, idx_y, _ = hand.landmarks[8]
        cmd = GestureCommand(move_point=(idx_x, idx_y))

        # Heuristicas de pinza
        d_thumb_index = _distance(norm[4], norm[8])
        d_thumb_middle = _distance(norm[4], norm[12])
        left_pinch = d_thumb_index < PINCH_THRESHOLD
        right_pinch = d_thumb_middle < PINCH_THRESHOLD

        now = time.time() * 1000  # ms

        if left_pinch and not self._left_pinch_active:
            self._left_pinch_active = True
            self._left_pinch_start = now

        if not left_pinch and self._left_pinch_active:
            duration = now - self._left_pinch_start
            if self._drag_active:
                cmd.drag_end = True
            elif duration < DRAG_HOLD_MS:
                if now - self._last_left_click_ms >= LEFT_CLICK_DEBOUNCE_MS:
                    cmd.click_left = True
                    self._last_left_click_ms = now
            self._left_pinch_active = False
            self._drag_active = False

        if self._left_pinch_active and not self._drag_active:
            duration = now - self._left_pinch_start
            if duration >= DRAG_HOLD_MS:
                self._drag_active = True
                cmd.drag_start = True

        if right_pinch and not self._right_pinch_active:
            # Click derecho al iniciar la pinza, con debounce para evitar repetidos.
            if now - self._last_right_click_ms >= RIGHT_CLICK_DEBOUNCE_MS:
                cmd.click_right = True
                self._last_right_click_ms = now
            self._right_pinch_active = True
        if not right_pinch and self._right_pinch_active:
            # Por si no se disparo en el inicio, se da otra oportunidad al soltar.
            if now - self._last_right_click_ms >= RIGHT_CLICK_DEBOUNCE_MS:
                cmd.click_right = True
                self._last_right_click_ms = now
            self._right_pinch_active = False

        # Scroll por doble dedo
        index_up = _finger_is_up(norm, 8, 6)
        middle_up = _finger_is_up(norm, 12, 10)
        ring_up = _finger_is_up(norm, 16, 14)
        pinky_up = _finger_is_up(norm, 20, 18)
        scroll_mode = index_up and middle_up and not ring_up and not pinky_up and not left_pinch

        if scroll_mode:
            current_y = norm[8][1]
            if self._scroll_prev_y is not None:
                delta = self._scroll_prev_y - current_y
                if abs(delta) > SCROLL_DEADZONE:
                    cmd.scroll = int(delta * SCROLL_GAIN)
            self._scroll_prev_y = current_y
        else:
            self._scroll_prev_y = None

        label = self._classify_with_model(norm)
        cmd.gesture_label = label
        # Opcional: mapear etiquetas de modelo a acciones.
        if label:
            lowered = label.lower()
            if "click_izquierdo" in lowered and now - self._last_left_click_ms >= LEFT_CLICK_DEBOUNCE_MS:
                cmd.click_left = True
                self._last_left_click_ms = now
            if "click_derecho" in lowered and now - self._last_right_click_ms >= RIGHT_CLICK_DEBOUNCE_MS:
                cmd.click_right = True
                self._last_right_click_ms = now
        return cmd
