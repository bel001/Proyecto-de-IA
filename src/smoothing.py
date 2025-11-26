from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple


class ExponentialSmoother:
    """
    Suaviza valores (x, y) usando media exponencial para reducir jitter.
    """

    def __init__(self, alpha: float = 0.25) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha debe estar en (0, 1].")
        self.alpha = alpha
        self._state: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._state = None

    def update(self, value: Tuple[float, float]) -> Tuple[float, float]:
        if self._state is None:
            self._state = value
            return value
        x, y = value
        prev_x, prev_y = self._state
        new_x = self.alpha * x + (1 - self.alpha) * prev_x
        new_y = self.alpha * y + (1 - self.alpha) * prev_y
        self._state = (new_x, new_y)
        return self._state


class MovingAverageSmoother:
    """
    Suaviza tomando la media de una ventana de los ultimos puntos.
    """

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window debe ser >= 1")
        self.window = window
        self._buf: Deque[Tuple[float, float]] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, value: Tuple[float, float]) -> Tuple[float, float]:
        self._buf.append(value)
        xs = [p[0] for p in self._buf]
        ys = [p[1] for p in self._buf]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


class OneEuroFilter:
    """
    Filtro One Euro: mantiene respuesta rapida reduciendo jitter segun la velocidad.
    Referencia: https://hal.science/hal-00670496/document
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.03, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._last_time: Optional[float] = None
        self._x_prev: Optional[Tuple[float, float]] = None
        self._dx_prev: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._last_time = None
        self._x_prev = None
        self._dx_prev = None

    @staticmethod
    def _alpha(cutoff: float, te: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    @staticmethod
    def _filter(x: float, x_prev: float, alpha: float) -> float:
        return alpha * x + (1 - alpha) * x_prev

    def update(self, value: Tuple[float, float], timestamp: Optional[float] = None) -> Tuple[float, float]:
        now = time.time() if timestamp is None else timestamp
        if self._last_time is None:
            self._last_time = now
            self._x_prev = value
            self._dx_prev = (0.0, 0.0)
            return value

        te = max(now - self._last_time, 1e-3)
        self._last_time = now

        dx = ((value[0] - self._x_prev[0]) / te, (value[1] - self._x_prev[1]) / te)
        alpha_d = self._alpha(self.d_cutoff, te)
        dx_hat = (
            self._filter(dx[0], self._dx_prev[0], alpha_d),
            self._filter(dx[1], self._dx_prev[1], alpha_d),
        )

        cutoff_x = self.min_cutoff + self.beta * abs(dx_hat[0])
        cutoff_y = self.min_cutoff + self.beta * abs(dx_hat[1])
        alpha_x = self._alpha(cutoff_x, te)
        alpha_y = self._alpha(cutoff_y, te)

        x_hat = self._filter(value[0], self._x_prev[0], alpha_x)
        y_hat = self._filter(value[1], self._x_prev[1], alpha_y)

        self._x_prev = (x_hat, y_hat)
        self._dx_prev = dx_hat
        return self._x_prev
