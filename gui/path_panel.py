from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .models import ControlPath


class PathPanel(QWidget):
    create_requested = Signal()
    active_path_requested = Signal(str)
    rename_requested = Signal(str, str)
    visibility_changed = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: dict[str, QTreeWidgetItem] = {}
        self._names: dict[str, str] = {}
        self._active_path_id: str | None = None

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["显示", "路径"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 48)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)

        self.create_button = QPushButton("新建路径", self)
        self.create_button.clicked.connect(self.create_requested)
        buttons = QHBoxLayout()
        buttons.addWidget(self.create_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.tree, 1)
        layout.addLayout(buttons)

    def set_paths(
        self,
        paths: Sequence[ControlPath],
        active_path_id: str | None,
    ) -> None:
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            self._items.clear()
            self._names.clear()
            self._active_path_id = active_path_id
            for path in paths:
                item = QTreeWidgetItem(self.tree)
                item.setData(0, Qt.ItemDataRole.UserRole, path.path_id)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked if path.visible else Qt.CheckState.Unchecked,
                )
                item.setText(1, path.name)
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEditable
                )
                if path.path_id == active_path_id:
                    font = item.font(1)
                    font.setBold(True)
                    item.setFont(1, font)
                    self.tree.setCurrentItem(item)
                self._items[path.path_id] = item
                self._names[path.path_id] = path.name
        finally:
            self.tree.blockSignals(False)

    def _path_id(self, item: QTreeWidgetItem) -> str:
        return str(item.data(0, Qt.ItemDataRole.UserRole))

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        path_id = self._path_id(item)
        if column == 1 and path_id != self._active_path_id:
            self.active_path_requested.emit(path_id)
        elif column == 0 and self._active_path_id in self._items:
            self.tree.setCurrentItem(self._items[self._active_path_id])

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        path_id = self._path_id(item)
        if column == 0:
            visible = item.checkState(0) == Qt.CheckState.Checked
            self.visibility_changed.emit(path_id, visible)
        elif column == 1:
            name = item.text(1).strip()
            if name and name != self._names.get(path_id):
                self._names[path_id] = name
                self.rename_requested.emit(path_id, name)

    def set_path_visibility(self, path_id: str, visible: bool) -> None:
        item = self._items[path_id]
        item.setCheckState(
            0,
            Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked,
        )

    def activate_path(self, path_id: str) -> None:
        if path_id not in self._items:
            raise KeyError(path_id)
        if path_id != self._active_path_id:
            self.active_path_requested.emit(path_id)

