from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable


class MouseIntent(Enum):
    NONE = auto()
    APPEND_POINT = auto()
    DRAG_POINT = auto()
    CONTEXT_MENU = auto()
    SLICE = auto()
    ZOOM = auto()
    PAN = auto()
    MAXIMIZE = auto()


@dataclass(frozen=True)
class HitTarget:
    path_id: str
    point_index: int
    is_active: bool


def classify_pointer_action(
    button: str,
    *,
    hit: HitTarget | None = None,
    ctrl: bool = False,
) -> MouseIntent:
    button = str(button).lower()
    if button == "wheel":
        return MouseIntent.ZOOM if ctrl else MouseIntent.SLICE
    if button == "middle":
        return MouseIntent.PAN
    if hit is not None and not hit.is_active:
        return MouseIntent.NONE
    if button == "left" and hit is not None:
        return MouseIntent.DRAG_POINT
    if button == "right" and hit is not None:
        return MouseIntent.CONTEXT_MENU
    if button == "left":
        return MouseIntent.APPEND_POINT
    return MouseIntent.NONE


class ClickDisambiguator:
    """Delay single-click work until Qt's double-click window has elapsed."""

    def __init__(
        self,
        interval_ms: int,
        schedule: Callable[[int, Callable[[], None]], Any],
        cancel: Callable[[Any], None],
    ) -> None:
        self.interval_ms = max(1, int(interval_ms))
        self._schedule = schedule
        self._cancel = cancel
        self._pending: Any | None = None

    @property
    def has_pending_click(self) -> bool:
        return self._pending is not None

    def cancel_pending(self) -> None:
        if self._pending is not None:
            self._cancel(self._pending)
            self._pending = None

    def release_empty(
        self,
        payload: Any,
        emit_append: Callable[[Any], None],
        emit_maximize: Callable[[], None],
    ) -> None:
        del emit_maximize
        self.cancel_pending()

        def fire() -> None:
            self._pending = None
            emit_append(payload)

        self._pending = self._schedule(self.interval_ms, fire)

    def double_click(self, emit_maximize: Callable[[], None]) -> None:
        self.cancel_pending()
        emit_maximize()

