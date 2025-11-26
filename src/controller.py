from __future__ import annotations

from typing import Tuple

import pyautogui

pyautogui.FAILSAFE = False


class CursorController:
    def __init__(self) -> None:
        self.screen_width, self.screen_height = pyautogui.size()
        self._dragging = False

    def move(self, point: Tuple[float, float]) -> None:
        # point viene en pixeles absolutos
        x = min(max(point[0], 0), self.screen_width - 1)
        y = min(max(point[1], 0), self.screen_height - 1)
        pyautogui.moveTo(x, y, duration=0)

    def click_left(self) -> None:
        pyautogui.click(button="left")

    def click_right(self) -> None:
        pyautogui.click(button="right")

    def drag_start(self) -> None:
        if not self._dragging:
            pyautogui.mouseDown(button="left")
            self._dragging = True

    def drag_end(self) -> None:
        if self._dragging:
            pyautogui.mouseUp(button="left")
            self._dragging = False

    def scroll(self, amount: int) -> None:
        if amount != 0:
            pyautogui.scroll(amount)
