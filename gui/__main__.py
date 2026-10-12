from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv if argv is None else argv)
    application = QApplication(arguments)
    application.setApplicationName("NarrowDetecter CPR")
    window = MainWindow()
    window.show()
    return int(application.exec())


if __name__ == "__main__":
    raise SystemExit(main())

