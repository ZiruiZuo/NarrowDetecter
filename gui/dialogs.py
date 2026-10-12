from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .image_io import DicomSeriesInfo


class DicomSeriesDialog(QDialog):
    def __init__(
        self,
        series: Sequence[DicomSeriesInfo],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择 DICOM 序列")
        self.resize(760, 360)
        self._series = tuple(series)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("检测到多个 DICOM 序列，请选择要载入的序列："))
        self.table = QTableWidget(len(self._series), 4, self)
        self.table.setHorizontalHeaderLabels(
            ["序列描述", "序列号", "层数", "Series UID"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row, info in enumerate(self._series):
            values = (
                info.description,
                info.series_number,
                str(info.file_count),
                info.series_uid,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, info.series_uid)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.table.itemSelectionChanged.connect(self._update_accept_enabled)
        self._update_accept_enabled()

    def _update_accept_enabled(self) -> None:
        button = self.button_box.button(QDialogButtonBox.Ok)
        button.setEnabled(self.table.currentRow() >= 0)

    def selected_series_uid(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        return self._series[row].series_uid
