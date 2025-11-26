from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


@dataclass
class HandDetection:
    landmarks: List[Tuple[float, float, float]]
    bbox: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax en coordenadas normalizadas
    handedness: str
    confidence: float


class HandTracker:
    """
    Envoltura ligera de MediaPipe Hands.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
    ) -> None:
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )

    def close(self) -> None:
        self._hands.close()

    def process(self, frame_bgr: np.ndarray) -> List[HandDetection]:
        # MediaPipe espera RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)
        detections: List[HandDetection] = []

        if not result.multi_hand_landmarks:
            return detections

        image_height, image_width = frame_bgr.shape[:2]

        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            landmarks = []
            xs, ys = [], []
            for lm in hand_landmarks.landmark:
                x, y, z = lm.x, lm.y, lm.z
                landmarks.append((x, y, z))
                xs.append(x)
                ys.append(y)

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            handedness = "unknown"
            confidence = 0.0
            if result.multi_handedness and idx < len(result.multi_handedness):
                handedness = result.multi_handedness[idx].classification[0].label
                confidence = result.multi_handedness[idx].classification[0].score

            detections.append(
                HandDetection(
                    landmarks=landmarks,
                    bbox=(xmin, ymin, xmax, ymax),
                    handedness=handedness,
                    confidence=confidence,
                )
            )
        return detections
