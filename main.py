"""
AI SecureVault — Entry Point
Run:  python main.py
"""

import sys
import os

# Make sure repo root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_logger

logger = get_logger("main")


def main():
    # ── 1. Initialise database ────────────────────────────────────────────────
    init_db()
    logger.info("AI SecureVault starting up.")

    # ── 2. Launch Qt application ──────────────────────────────────────────────
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from gui import MainWindow

    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app    = QApplication(sys.argv)
    app.setApplicationName("AI SecureVault")
    app.setOrganizationName("SecureVault")

    window = MainWindow()
    window.show()

    exit_code = app.exec_()

    logger.info("AI SecureVault shut down (exit code %d).", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
