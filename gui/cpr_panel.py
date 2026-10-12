from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .models import CPRSettings


class CPRPanel(QWidget):
    start_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode = QComboBox(self)
        self.mode.addItem("Stretched", "stretched")
        self.mode.addItem("Straightened", "straightened")
        self.angle_start = self._spin(-3600.0, 3600.0, 0.0, 1.0)
        self.angle_end = self._spin(-3600.0, 3600.0, 165.0, 1.0)
        self.angle_step = self._spin(-360.0, 360.0, 15.0, 1.0)
        self.sampling_width = self._spin(0.01, 1000.0, 40.0, 1.0)
        self.slice_resolution = self._spin(0.01, 20.0, 1.0, 0.1)
        self.show_centerline = QCheckBox("在导出图像上显示中心线", self)
        self.show_centerline.setChecked(True)

        form = QFormLayout()
        form.addRow("模式", self.mode)
        form.addRow("起始角度 (°)", self.angle_start)
        form.addRow("终止角度 (°)", self.angle_end)
        form.addRow("角度步长 (°)", self.angle_step)
        form.addRow("采样宽度 (mm)", self.sampling_width)
        form.addRow("输出分辨率 (mm)", self.slice_resolution)
        form.addRow(self.show_centerline)

        self.output_edit = QLineEdit(self)
        self.output_button = QPushButton("选择…", self)
        self.output_button.clicked.connect(self._choose_output_directory)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_button)
        form.addRow("输出目录", output_row)

        self.validation_label = QLabel(self)
        self.validation_label.setStyleSheet("color: #d9534f")
        self.validation_label.setWordWrap(True)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label = QLabel("就绪", self)
        self.start_button = QPushButton("Start", self)
        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested)
        self.cancel_button.clicked.connect(self.cancel_requested)
        actions = QHBoxLayout()
        actions.addWidget(self.start_button)
        actions.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(form)
        layout.addWidget(self.validation_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addLayout(actions)
        layout.addStretch(1)

    def _spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def settings(self) -> CPRSettings:
        settings = CPRSettings(
            mode=str(self.mode.currentData()),
            angle_start_degrees=self.angle_start.value(),
            angle_end_degrees=self.angle_end.value(),
            angle_step_degrees=self.angle_step.value(),
            sampling_line_length_mm=self.sampling_width.value(),
            slice_resolution_mm=self.slice_resolution.value(),
            show_centerline=self.show_centerline.isChecked(),
        )
        try:
            settings.validate()
        except ValueError as error:
            self.validation_label.setText(str(error))
            raise
        self.validation_label.clear()
        return settings

    def set_settings(self, settings: CPRSettings) -> None:
        settings.validate()
        index = self.mode.findData(str(settings.mode).lower())
        self.mode.setCurrentIndex(index)
        self.angle_start.setValue(settings.angle_start_degrees)
        self.angle_end.setValue(settings.angle_end_degrees)
        self.angle_step.setValue(settings.angle_step_degrees)
        self.sampling_width.setValue(settings.sampling_line_length_mm)
        self.slice_resolution.setValue(settings.slice_resolution_mm)
        self.show_centerline.setChecked(settings.show_centerline)
        self.validation_label.clear()

    def output_directory(self) -> Path | None:
        value = self.output_edit.text().strip()
        return Path(value) if value else None

    def set_output_directory(self, path: str | Path) -> None:
        self.output_edit.setText(str(path))

    def _choose_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择 CPR 输出目录",
            self.output_edit.text().strip(),
        )
        if directory:
            self.output_edit.setText(directory)

    def set_busy(self, busy: bool) -> None:
        busy = bool(busy)
        for widget in (
            self.mode,
            self.angle_start,
            self.angle_end,
            self.angle_step,
            self.sampling_width,
            self.slice_resolution,
            self.show_centerline,
            self.output_edit,
            self.output_button,
        ):
            widget.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, int(total)))
        self.progress.setValue(int(done))

    def set_status(self, text: str) -> None:
        self.status_label.setText(str(text))

