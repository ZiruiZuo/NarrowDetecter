from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from .cpr_adapter import (
    CPRCancelled,
    ManualCPRRequest,
    export_manual_cpr,
)


class CPRWorker(QObject):
    progress = Signal(int, int)
    completed = Signal(str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, request: ManualCPRRequest) -> None:
        super().__init__()
        self._request = request
        self._cancel_event = Event()

    @Slot()
    def run(self) -> None:
        try:
            output = export_manual_cpr(
                self._request,
                is_cancelled=self._cancel_event.is_set,
                on_progress=self.progress.emit,
            )
        except CPRCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(str(output))

    @Slot()
    def cancel(self) -> None:
        self._cancel_event.set()
