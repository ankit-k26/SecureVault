"""
AI SecureVault - PyQt5 GUI
===========================
Screens:
  1. LoginScreen      — live camera feed + password fallback
  2. DashboardScreen  — vault controls, navigation
  3. FilesBrowserScreen — vault file listing
  4. IntruderLogsScreen — captured intruder images + timestamps
  5. SettingsScreen   — user management, face enrolment, change password
  6. SetupWizard      — first-run user creation + face enrolment
"""

from __future__ import annotations

import os
import sys
import getpass
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QMimeData, QUrl,
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QIcon,
    QDragEnterEvent, QDropEvent,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit,
    QFrame, QScrollArea, QFileDialog,
    QMessageBox, QInputDialog, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QSpacerItem,
)

from config import (
    APP_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    THEME_PRIMARY, THEME_ACCENT, THEME_SURFACE, THEME_TEXT,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_SKIP,
    INTRUDER_DIR,CAMERA_FPS
)
from auth import AuthController, AuthState, get_controller
from database import get_auth_events, get_intruder_logs, list_users, clear_all_intruder_logs


# ─────────────────────────────────────────────────────────────────────────────
# Stylesheet
# ─────────────────────────────────────────────────────────────────────────────

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {THEME_PRIMARY};
    color: {THEME_TEXT};
    font-family: 'Segoe UI', 'Arial', sans-serif;
}}
QPushButton {{
    background-color: {THEME_SURFACE};
    color: {THEME_TEXT};
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {THEME_ACCENT};
    border-color: {THEME_ACCENT};
}}
QPushButton:pressed {{
    background-color: #c73450;
}}
QPushButton#accent {{
    background-color: {THEME_ACCENT};
    font-weight: bold;
    font-size: 14px;
    padding: 10px 24px;
}}
QPushButton#accent:hover {{
    background-color: #ff5577;
}}
QPushButton#danger {{
    background-color: #8b1a1a;
    border-color: #c0392b;
}}
QPushButton#danger:hover {{
    background-color: #c0392b;
}}
QLineEdit {{
    background-color: {THEME_SURFACE};
    color: {THEME_TEXT};
    border: 1px solid #2a2a4a;
    border-radius: 5px;
    padding: 8px 12px;
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {THEME_ACCENT};
}}
QLabel {{
    color: {THEME_TEXT};
}}
QLabel#title {{
    font-size: 22px;
    font-weight: bold;
    color: {THEME_ACCENT};
}}
QLabel#subtitle {{
    font-size: 13px;
    color: #aaaacc;
}}
QFrame#card {{
    background-color: {THEME_SURFACE};
    border: 1px solid #2a2a4a;
    border-radius: 10px;
}}
QTableWidget {{
    background-color: {THEME_SURFACE};
    color: {THEME_TEXT};
    border: 1px solid #2a2a4a;
    gridline-color: #2a2a4a;
    selection-background-color: {THEME_ACCENT};
}}
QHeaderView::section {{
    background-color: {THEME_PRIMARY};
    color: {THEME_TEXT};
    border: 1px solid #2a2a4a;
    padding: 4px;
}}
QScrollBar:vertical {{
    background: {THEME_SURFACE};
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: #3a3a6a;
    border-radius: 4px;
}}
QProgressBar {{
    background-color: {THEME_SURFACE};
    color: {THEME_TEXT};
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {THEME_ACCENT};
    border-radius: 3px;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Camera worker thread
# ─────────────────────────────────────────────────────────────────────────────

class CameraWorker(QThread):
    """
    Runs the webcam capture loop in a background thread.
    Emits processed frames and recognition results to the GUI.
    """
    frame_ready   = pyqtSignal(QImage)       # annotated frame for display
    auth_result   = pyqtSignal(object, object)  # (RecognitionResult, AuthState)

    def __init__(self, ctrl: AuthController, parent=None):
        super().__init__(parent)
        self._ctrl    = ctrl
        self._running = False
        self._frame_n = 0

    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                self.msleep(30)
                continue

            # ── Guard: ensure frame is a valid 8-bit BGR image ───────────────
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.msleep(30)
                continue

            self._frame_n += 1
            result  = None
            state   = self._ctrl.state

            # Only run the heavy recognition every FRAME_SKIP frames
            if (self._frame_n % FRAME_SKIP == 0 and
                    state == AuthState.SCANNING_FACE):
                from face_module import RecognitionResult
                result, state = self._ctrl.process_face_frame(frame)

            # Annotate — pass a uint8 copy, skip re-detect by passing locations=[]
            if result is not None:
                from face_module import FaceManager
                try:
                    annotated = FaceManager.annotate_frame(frame, result, locations=None)
                except Exception:
                    annotated = frame
            else:
                annotated = frame

            # Convert BGR → QImage
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)

            if result is not None:
                self.auth_result.emit(result, state)

            self.msleep(1000 // 30)   # ~30 fps

        cap.release()

    def stop(self) -> None:
        self._running = False
        self.wait(3000)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable widget helpers
# ─────────────────────────────────────────────────────────────────────────────

def _card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("card")
    f.setLayout(QVBoxLayout())
    f.layout().setContentsMargins(16, 16, 16, 16)
    f.layout().setSpacing(10)
    return f


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #2a2a4a;")
    return line


def _status_label(text: str = "", color: str = THEME_TEXT) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# Screen 1 — Login
# ─────────────────────────────────────────────────────────────────────────────

class LoginScreen(QWidget):
    authenticated = pyqtSignal(str)   # emits username on success

    def __init__(self, ctrl: AuthController, parent=None):
        super().__init__(parent)
        self._ctrl       = ctrl
        self._worker: Optional[CameraWorker] = None
        self._pwd_mode   = False
        self._password   = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QFrame()
        hdr.setStyleSheet(f"background-color: {THEME_SURFACE}; border-bottom: 2px solid {THEME_ACCENT};")
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(20, 12, 20, 12)
        title = QLabel("🔐  " + APP_TITLE)
        title.setObjectName("title")
        hdr_l.addWidget(title)
        hdr_l.addStretch()
        root.addWidget(hdr)

        # ── Body ────────────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(30, 20, 30, 20)
        body.setSpacing(20)
        root.addLayout(body)

        # Camera pane
        cam_card = _card()
        cam_card.setMinimumWidth(440)
        self._cam_label = QLabel("Camera initialising…")
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(420, 315)
        self._cam_label.setStyleSheet("background:#0a0a1a; border-radius:6px;")
        cam_card.layout().addWidget(self._cam_label)

        self._face_status = _status_label("Scanning for face…", "#aaaacc")
        cam_card.layout().addWidget(self._face_status)

        self._liveness_bar = QProgressBar()
        self._liveness_bar.setMaximum(100)
        self._liveness_bar.setValue(0)
        self._liveness_bar.setFormat("Liveness: %p%")
        self._liveness_bar.setMaximumHeight(18)
        cam_card.layout().addWidget(self._liveness_bar)
        body.addWidget(cam_card)

        # Auth pane
        auth_card = _card()
        auth_card.setMinimumWidth(280)

        vault_icon = QLabel("🛡️")
        vault_icon.setAlignment(Qt.AlignCenter)
        vault_icon.setStyleSheet("font-size: 56px;")
        auth_card.layout().addWidget(vault_icon)

        self._auth_title = QLabel("Face Recognition")
        self._auth_title.setObjectName("title")
        self._auth_title.setAlignment(Qt.AlignCenter)
        auth_card.layout().addWidget(self._auth_title)

        self._auth_hint = QLabel("Position your face in front of the camera.\nBlink to verify liveness.")
        self._auth_hint.setObjectName("subtitle")
        self._auth_hint.setAlignment(Qt.AlignCenter)
        self._auth_hint.setWordWrap(True)
        auth_card.layout().addWidget(self._auth_hint)

        auth_card.layout().addWidget(_hline())

        # Password fallback (hidden until needed)
        self._pwd_frame = QFrame()
        pwd_l = QVBoxLayout(self._pwd_frame)
        pwd_l.setContentsMargins(0, 0, 0, 0)
        pwd_l.setSpacing(8)

        self._username_input = QLineEdit()
        self._username_input.setPlaceholderText("Username")
        pwd_l.addWidget(self._username_input)

        self._password_input = QLineEdit()
        self._password_input.setPlaceholderText("Password")
        self._password_input.setEchoMode(QLineEdit.Password)
        self._password_input.returnPressed.connect(self._try_password)
        pwd_l.addWidget(self._password_input)

        self._login_btn = QPushButton("Unlock")
        self._login_btn.setObjectName("accent")
        self._login_btn.clicked.connect(self._try_password)
        pwd_l.addWidget(self._login_btn)

        auth_card.layout().addWidget(self._pwd_frame)
        self._pwd_frame.hide()

        auth_card.layout().addStretch()

        self._switch_btn = QPushButton("Use Password Instead")
        self._switch_btn.clicked.connect(self._toggle_mode)
        auth_card.layout().addWidget(self._switch_btn)

        self._msg_label = _status_label("", THEME_ACCENT)
        auth_card.layout().addWidget(self._msg_label)

        body.addWidget(auth_card)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start camera + face auth."""
        self._ctrl.start_face_auth()
        self._worker = CameraWorker(self._ctrl)
        self._worker.frame_ready.connect(self._update_frame)
        self._worker.auth_result.connect(self._on_auth_result)
        self._worker.start()

    def stop(self):
        if self._worker:
            self._worker.stop()
            self._worker = None

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _update_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self._cam_label.setPixmap(
            pix.scaled(self._cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_auth_result(self, result, state):
        from face_module import RecognitionResult

        # Ignore any stale camera signals once we've moved to password mode
        if self._pwd_mode:
            return

        if state == AuthState.AUTHENTICATED:
            self._face_status.setText(f"✅  Welcome, {result.username}!")
            self._face_status.setStyleSheet("color: #00e676; font-size: 14px; font-weight: bold;")
            self.stop()
            self.authenticated.emit(result.username)

        elif state == AuthState.AWAIT_PASSWORD:
            self._face_status.setText("⚠️  Face not recognised — enter password.")
            self._face_status.setStyleSheet(f"color: {THEME_ACCENT}; font-size: 13px;")
            self._show_password_mode()

        elif state == AuthState.LOCKED_OUT:
            secs = int(self._ctrl.lockout_remaining)
            self._face_status.setText(f"🔒  Locked out. Try again in {secs}s")
            self._face_status.setStyleSheet(f"color: {THEME_ACCENT};")

        else:
            if result.face_found:
                conf_pct = int(result.confidence * 100)
                self._liveness_bar.setValue(conf_pct)
                if result.matched:
                    self._face_status.setText(f"🔍  Matching… conf={conf_pct}%")
                    self._face_status.setStyleSheet("color: #ffeb3b;")
                else:
                    self._face_status.setText("❌  Unknown face")
                    self._face_status.setStyleSheet(f"color: {THEME_ACCENT};")
            else:
                self._liveness_bar.setValue(0)
                self._face_status.setText("Scanning for face…")
                self._face_status.setStyleSheet("color: #aaaacc;")

    def _show_password_mode(self):
        self._pwd_mode = True
        # Stop the camera worker — we no longer need face scanning and its
        # stale signals must not re-disable the Unlock button.
        self.stop()
        self._pwd_frame.show()
        self._auth_title.setText("Password Login")
        self._auth_hint.setText("Face recognition failed.\nEnter your credentials to unlock.")
        self._switch_btn.setText("Try Face Again")
        # Ensure Unlock button is always enabled when the form first appears
        self._login_btn.setEnabled(True)

    def _toggle_mode(self):
        if self._pwd_mode:
            self._pwd_mode = False
            self._pwd_frame.hide()
            self._auth_title.setText("Face Recognition")
            self._auth_hint.setText("Position your face in front of the camera.\nBlink to verify liveness.")
            self._switch_btn.setText("Use Password Instead")
            self._ctrl.start_face_auth()
            if not self._worker or not self._worker.isRunning():
                self.start()
        else:
            self._show_password_mode()

    def _try_password(self):
        username = self._username_input.text().strip()
        password = self._password_input.text()
        if not username or not password:
            self._msg_label.setText("Enter username and password.")
            return

        self._login_btn.setEnabled(False)
        result = self._ctrl.authenticate_password(username, password)

        if result.success:
            self._msg_label.setText("✅  Authenticated!")
            self.stop()
            self.authenticated.emit(username)
        else:
            self._msg_label.setText(result.message)
            self._login_btn.setEnabled(True)
            self._password_input.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Screen 2 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class DashboardScreen(QWidget):
    navigate = pyqtSignal(str)    # emits page key: 'files','logs','settings'
    lock_req = pyqtSignal()

    def __init__(self, ctrl: AuthController, parent=None):
        super().__init__(parent)
        self._ctrl = ctrl
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 20, 30, 20)
        root.setSpacing(16)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("Dashboard")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()
        self._user_label = QLabel()
        self._user_label.setObjectName("subtitle")
        hdr.addWidget(self._user_label)
        root.addLayout(hdr)
        root.addWidget(_hline())

        # Vault status card
        status_card = _card()
        status_row  = QHBoxLayout()

        self._vault_icon   = QLabel("🔒")
        self._vault_icon.setStyleSheet("font-size: 42px;")
        self._vault_status = QLabel("Locked")
        self._vault_status.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {THEME_ACCENT};")
        status_row.addWidget(self._vault_icon)
        status_row.addWidget(self._vault_status)
        status_row.addStretch()

        self._unlock_btn = QPushButton("🔓  Unlock Vault")
        self._unlock_btn.setObjectName("accent")
        self._unlock_btn.clicked.connect(self._toggle_vault)
        status_row.addWidget(self._unlock_btn)

        status_card.layout().addLayout(status_row)
        root.addWidget(status_card)

        # Action grid
        grid = QGridLayout()
        grid.setSpacing(14)

        actions = [
            ("📁  Browse Files",    "files",    "View and manage files in the vault."),
            ("👁️  Intruder Logs",   "logs",     "View captured intruder images."),
            ("⚙️  Settings",        "settings", "Manage users, faces, and security."),
            ("📊  Auth History",    "history",  "Review authentication events."),
        ]
        for i, (label, key, tip) in enumerate(actions):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.setMinimumHeight(70)
            btn.clicked.connect(lambda _, k=key: self.navigate.emit(k))
            grid.addWidget(btn, i // 2, i % 2)

        root.addLayout(grid)
        root.addStretch()

        # Lock button
        lock_btn = QPushButton("🔒  Lock & Sign Out")
        lock_btn.setObjectName("danger")
        lock_btn.clicked.connect(self.lock_req.emit)
        root.addWidget(lock_btn)

    def refresh(self):
        """Called when this screen becomes active."""
        user = self._ctrl.current_user or "Unknown"
        self._user_label.setText(f"Signed in as: {user}")
        self._update_vault_status()

    def _update_vault_status(self):
        unlocked = self._ctrl.vault.is_unlocked
        self._vault_icon.setText("🔓" if unlocked else "🔒")
        self._vault_status.setText("Unlocked" if unlocked else "Locked")
        self._vault_status.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {'#00e676' if unlocked else THEME_ACCENT};"
        )
        self._unlock_btn.setText("🔒  Lock Vault" if unlocked else "🔓  Unlock Vault")

    def _toggle_vault(self):
        if self._ctrl.vault.is_unlocked:
            self._ctrl.vault.lock(username=self._ctrl.current_user)
            QMessageBox.information(self, "Vault Locked", "All files have been re-encrypted.")
        else:
            password, ok = QInputDialog.getText(
                self, "Unlock Vault", "Enter your password:", QLineEdit.Password
            )
            if ok and password:
                if self._ctrl.post_auth_unlock(password):
                    QMessageBox.information(self, "Vault Unlocked", "Files are now accessible.")
                else:
                    QMessageBox.critical(self, "Error", "Wrong password — vault remains locked.")
        self._update_vault_status()


# ─────────────────────────────────────────────────────────────────────────────
# Screen 3 — Files Browser
# ─────────────────────────────────────────────────────────────────────────────

class FilesBrowserScreen(QWidget):
    def __init__(self, ctrl: AuthController, parent=None):
        super().__init__(parent)
        self._ctrl = ctrl
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)

        hdr = QHBoxLayout()
        title = QLabel("Vault Files")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()

        add_btn = QPushButton("➕  Add File")
        add_btn.clicked.connect(self._add_file)
        hdr.addWidget(add_btn)

        refresh_btn = QPushButton("🔄  Refresh")
        refresh_btn.clicked.connect(self.refresh)
        hdr.addWidget(refresh_btn)

        root.addLayout(hdr)
        root.addWidget(_hline())

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Name", "Size", "Status", "Modified"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self._table)

        # Action buttons
        btn_row = QHBoxLayout()
        open_btn = QPushButton("📂  Open in Explorer")
        open_btn.clicked.connect(self._open_folder)
        del_btn = QPushButton("🗑️  Delete Selected")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        btn_row.addWidget(del_btn)
        root.addLayout(btn_row)

        self._status = QLabel("Vault is locked.")
        self._status.setObjectName("subtitle")
        root.addWidget(self._status)

    def refresh(self):
        files = self._ctrl.vault.list_files()
        self._table.setRowCount(len(files))
        for row, f in enumerate(files):
            self._table.setItem(row, 0, QTableWidgetItem(f["name"]))
            size_str = f"{f['size'] / 1024:.1f} KB" if f["size"] > 1024 else f"{f['size']} B"
            self._table.setItem(row, 1, QTableWidgetItem(size_str))
            enc_str  = "🔒 Encrypted" if f["encrypted"] else "🔓 Plaintext"
            self._table.setItem(row, 2, QTableWidgetItem(enc_str))
            mod = datetime.fromtimestamp(f["modified"]).strftime("%Y-%m-%d %H:%M")
            self._table.setItem(row, 3, QTableWidgetItem(mod))

        vault_state = "Unlocked 🔓" if self._ctrl.vault.is_unlocked else "Locked 🔒"
        self._status.setText(f"Vault: {vault_state}  |  {len(files)} file(s)")

    def _add_file(self):
        if not self._ctrl.vault.is_unlocked:
            QMessageBox.warning(self, "Locked", "Unlock the vault before adding files.")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Select files to add")
        for path in paths:
            self._ctrl.vault.add_file(path, username=self._ctrl.current_user)
        self.refresh()

    def _open_folder(self):
        import subprocess, platform
        path = self._ctrl.vault.protected_dir
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _delete_selected(self):
        rows = set(i.row() for i in self._table.selectedItems())
        if not rows:
            return
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Permanently delete {len(rows)} file(s)?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        files = self._ctrl.vault.list_files()
        for row in sorted(rows, reverse=True):
            if row < len(files):
                self._ctrl.vault.delete_file(files[row]["path"])
        self.refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Intruder Image Viewer — full-size popup dialog
# ─────────────────────────────────────────────────────────────────────────────

class IntruderImageViewer(QWidget):
    """
    Full-size image viewer dialog for a single intruder capture.
    Opens as a standalone window; multiple can be open at once.
    """

    def __init__(self, image_path: str, timestamp: str, confidence: float | None, parent=None):
        super().__init__(parent, Qt.Window)
        self._path       = image_path
        self._timestamp  = timestamp
        self._confidence = confidence
        self._original_pixmap: QPixmap | None = None
        self.setWindowTitle(f"Intruder — {timestamp[:16]}")
        self.setMinimumSize(520, 480)
        self.setStyleSheet(f"""
            QWidget  {{ background-color: {THEME_PRIMARY}; color: {THEME_TEXT}; }}
            QLabel   {{ color: {THEME_TEXT}; }}
            QPushButton {{
                background-color: {THEME_SURFACE};
                color: {THEME_TEXT};
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                padding: 7px 18px;
                font-size: 13px;
            }}
            QPushButton:hover {{ background-color: {THEME_ACCENT}; border-color: {THEME_ACCENT}; }}
        """)
        self._build_ui()
        self._load_image()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # ── Image display ─────────────────────────────────────────────────────
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet("background:#0a0a1a; border-radius:6px;")
        self._img_label.setMinimumSize(480, 360)
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._img_label)

        # ── Metadata row ──────────────────────────────────────────────────────
        meta_row = QHBoxLayout()

        ts_label = QLabel(f"🕐  {self._timestamp[:19].replace('T', '  ')}")
        ts_label.setStyleSheet("font-size: 13px; color: #aaaacc;")
        meta_row.addWidget(ts_label)

        meta_row.addStretch()

        if self._confidence is not None:
            conf_label = QLabel(f"Confidence: {self._confidence:.1%}")
            conf_label.setStyleSheet("font-size: 13px; color: #aaaacc;")
            meta_row.addWidget(conf_label)

        root.addLayout(meta_row)

        path_label = QLabel(f"📂  {self._path}")
        path_label.setStyleSheet("font-size: 11px; color: #666688;")
        path_label.setWordWrap(True)
        root.addWidget(path_label)

        root.addWidget(_hline())

        # ── Action buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        open_btn = QPushButton("📂  Open in File Manager")
        open_btn.clicked.connect(self._open_in_file_manager)
        btn_row.addWidget(open_btn)

        save_btn = QPushButton("💾  Save Copy As…")
        save_btn.clicked.connect(self._save_copy)
        btn_row.addWidget(save_btn)

        btn_row.addStretch()

        close_btn = QPushButton("✖  Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

    def _load_image(self):
        if os.path.exists(self._path):
            self._original_pixmap = QPixmap(self._path)
            self._fit_image()
        else:
            self._img_label.setText("⚠️  Image file not found on disk.")

    def _fit_image(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self._img_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        """Re-scale image whenever the window is resized."""
        super().resizeEvent(event)
        self._fit_image()

    def _open_in_file_manager(self):
        import subprocess, platform
        folder = os.path.dirname(self._path)
        try:
            if platform.system() == "Windows":
                # /select highlights the specific file in Explorer
                subprocess.Popen(["explorer", "/select,", self._path])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-R", self._path])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Could not open file manager:\n{exc}")

    def _save_copy(self):
        if not self._original_pixmap or self._original_pixmap.isNull():
            QMessageBox.warning(self, "No Image", "No image loaded to save.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Intruder Image",
            os.path.basename(self._path),
            "Images (*.jpg *.jpeg *.png *.bmp)",
        )
        if dest:
            if self._original_pixmap.save(dest):
                QMessageBox.information(self, "Saved", f"Image saved to:\n{dest}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save the image.")


# ─────────────────────────────────────────────────────────────────────────────
# Screen 4 — Intruder Logs
# ─────────────────────────────────────────────────────────────────────────────

class IntruderLogsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewers: list[IntruderImageViewer] = []   # keep refs alive
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)

        hdr = QHBoxLayout()
        title = QLabel("Intruder Logs")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()

        refresh_btn = QPushButton("🔄  Refresh")
        refresh_btn.clicked.connect(self.refresh)
        hdr.addWidget(refresh_btn)

        clear_btn = QPushButton("🗑️  Clear All")
        clear_btn.setObjectName("danger")
        clear_btn.clicked.connect(self._clear_all)
        hdr.addWidget(clear_btn)

        root.addLayout(hdr)
        root.addWidget(_hline())

        # Scroll area for intruder image cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(12)
        scroll.setWidget(self._grid_widget)
        root.addWidget(scroll)

        self._count_label = QLabel("No intruder events recorded.")
        self._count_label.setObjectName("subtitle")
        root.addWidget(self._count_label)

    def refresh(self):
        # Close any open viewer windows
        for v in self._viewers:
            v.close()
        self._viewers.clear()

        # Clear grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        logs = get_intruder_logs(50)
        self._count_label.setText(f"{len(logs)} intruder event(s) captured.")

        for idx, log in enumerate(logs):
            card = _card()
            card.setMaximumWidth(200)

            # ── Thumbnail (clickable) ─────────────────────────────────────────
            img_label = QLabel()
            img_label.setFixedSize(170, 128)
            img_label.setStyleSheet(
                "background:#0a0a1a; border-radius:4px;"
                "border: 2px solid transparent;"
            )
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setCursor(Qt.PointingHandCursor)
            img_label.setToolTip("Double-click to open full-size")

            path = log.get("image_path", "")
            if os.path.exists(path):
                pix = QPixmap(path)
                img_label.setPixmap(
                    pix.scaled(170, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                img_label.setText("Image\nnot found")

            # Capture loop variables for the lambda
            _path = path
            _ts   = log.get("timestamp", "")
            _conf = log.get("confidence")

            # Double-click on thumbnail opens viewer
            img_label.mouseDoubleClickEvent = lambda _e, p=_path, t=_ts, c=_conf: (
                self._open_viewer(p, t, c)
            )

            card.layout().addWidget(img_label)

            ts_label = QLabel(_ts[:16])
            ts_label.setObjectName("subtitle")
            ts_label.setAlignment(Qt.AlignCenter)
            card.layout().addWidget(ts_label)

            if _conf is not None:
                conf_label = QLabel(f"Conf: {_conf:.1%}")
                conf_label.setObjectName("subtitle")
                conf_label.setAlignment(Qt.AlignCenter)
                card.layout().addWidget(conf_label)

            # ── Open button ───────────────────────────────────────────────────
            open_btn = QPushButton("🔍  Open")
            open_btn.setToolTip("Open full-size image viewer")
            open_btn.clicked.connect(
                lambda _checked, p=_path, t=_ts, c=_conf: self._open_viewer(p, t, c)
            )
            card.layout().addWidget(open_btn)

            row, col = divmod(idx, 4)
            self._grid.addWidget(card, row, col)

    def _open_viewer(self, path: str, timestamp: str, confidence: float | None):
        """Open a full-size viewer window for the given intruder image."""
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Not Found", f"Image file not found:\n{path}")
            return
        viewer = IntruderImageViewer(path, timestamp, confidence, parent=None)
        self._viewers.append(viewer)
        # Remove from list when the viewer is closed so we don't accumulate refs
        viewer.destroyed.connect(lambda: self._viewers.remove(viewer) if viewer in self._viewers else None)
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()

    def _clear_all(self):
        reply = QMessageBox.question(
            self, "Clear All Intruder Data",
            "Permanently delete all intruder images and records?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            count = clear_all_intruder_logs()
            QMessageBox.information(
                self, "Cleared",
                f"Deleted {count} intruder record(s) and their images."
            )
            self.refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Screen 5 — Settings
# ─────────────────────────────────────────────────────────────────────────────

class SettingsScreen(QWidget):
    def __init__(self, ctrl: AuthController, parent=None):
        super().__init__(parent)
        self._ctrl = ctrl
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)

        title = QLabel("Settings")
        title.setObjectName("title")
        root.addWidget(title)
        root.addWidget(_hline())

        # ── User management ───────────────────────────────────────────────────
        users_card = _card()
        users_card.layout().addWidget(QLabel("👤  User Management"))

        self._users_table = QTableWidget(0, 3)
        self._users_table.setHorizontalHeaderLabels(["Username", "Created", "Active"])
        self._users_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._users_table.setMaximumHeight(160)
        users_card.layout().addWidget(self._users_table)

        btn_row = QHBoxLayout()
        add_user_btn = QPushButton("➕ Add User")
        add_user_btn.clicked.connect(self._add_user)
        remove_user_btn = QPushButton("🗑️ Remove User")
        remove_user_btn.setObjectName("danger")
        remove_user_btn.clicked.connect(self._remove_user)
        btn_row.addWidget(add_user_btn)
        btn_row.addWidget(remove_user_btn)
        btn_row.addStretch()
        users_card.layout().addLayout(btn_row)
        root.addWidget(users_card)

        # ── Face management ───────────────────────────────────────────────────
        face_card = _card()
        face_card.layout().addWidget(QLabel("📸  Face Registration"))

        self._face_list = QLabel("No faces registered.")
        self._face_list.setObjectName("subtitle")
        face_card.layout().addWidget(self._face_list)

        face_btn_row = QHBoxLayout()
        enroll_btn = QPushButton("📷  Enrol Face (Webcam)")
        enroll_btn.clicked.connect(self._enroll_face)
        remove_face_btn = QPushButton("🗑️  Remove Face")
        remove_face_btn.setObjectName("danger")
        remove_face_btn.clicked.connect(self._remove_face)
        face_btn_row.addWidget(enroll_btn)
        face_btn_row.addWidget(remove_face_btn)
        face_btn_row.addStretch()
        face_card.layout().addLayout(face_btn_row)
        root.addWidget(face_card)

        # ── Password change ───────────────────────────────────────────────────
        pwd_card = _card()
        pwd_card.layout().addWidget(QLabel("🔑  Change Password"))

        pwd_form = QHBoxLayout()
        self._new_pwd_input = QLineEdit()
        self._new_pwd_input.setPlaceholderText("New password")
        self._new_pwd_input.setEchoMode(QLineEdit.Password)
        change_pwd_btn = QPushButton("Update")
        change_pwd_btn.clicked.connect(self._change_password)
        pwd_form.addWidget(self._new_pwd_input)
        pwd_form.addWidget(change_pwd_btn)
        pwd_card.layout().addLayout(pwd_form)
        root.addWidget(pwd_card)

        root.addStretch()

    def refresh(self):
        # Users table
        users = list_users()
        self._users_table.setRowCount(len(users))
        for row, u in enumerate(users):
            self._users_table.setItem(row, 0, QTableWidgetItem(u["username"]))
            self._users_table.setItem(row, 1, QTableWidgetItem(u["created_at"][:16]))
            self._users_table.setItem(row, 2, QTableWidgetItem("✅" if u["is_active"] else "❌"))

        # Face list
        faces = self._ctrl.face_manager.list_registered_users()
        self._face_list.setText("Registered faces: " + (", ".join(faces) if faces else "none"))

    def _add_user(self):
        username, ok = QInputDialog.getText(self, "Add User", "Username:")
        if not ok or not username:
            return
        password, ok = QInputDialog.getText(self, "Add User", "Password:", QLineEdit.Password)
        if not ok or not password:
            return
        if self._ctrl.user_manager.create_user(username, password):
            QMessageBox.information(self, "User Created", f"'{username}' added successfully.")
        else:
            QMessageBox.warning(self, "Error", f"Username '{username}' already exists.")
        self.refresh()

    def _remove_user(self):
        row = self._users_table.currentRow()
        if row < 0:
            return
        username = self._users_table.item(row, 0).text()
        reply = QMessageBox.question(
            self, "Remove User",
            f"Remove user '{username}' and their face data?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._ctrl.user_manager.remove_user(username)
            self.refresh()

    def _enroll_face(self):
        username, ok = QInputDialog.getText(
            self, "Enrol Face", "Username to enrol:",
            text=self._ctrl.current_user or ""
        )
        if not ok or not username:
            return

        cap = cv2.VideoCapture(CAMERA_INDEX)
        frames_captured = 0
        captured_paths  = []
        import uuid, tempfile

        QMessageBox.information(
            self, "Face Enrolment",
            "Press SPACE to capture (5 shots recommended), then close the window."
        )

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Reject non-standard frames (wrong dtype or channel count)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                continue

            done = frames_captured >= 10
            hint = "(max reached, press Q)" if done else "SPACE=capture  Q=done"
            cv2.putText(frame, f"Captured: {frames_captured}/10  {hint}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Enrol Face — SPACE to capture, Q to finish", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and not done:
                # Use .copy() to ensure a fresh memory buffer for the save operation
                frame_to_save = frame.copy()

                tmp = os.path.join(tempfile.gettempdir(), f"enrol_{uuid.uuid4().hex}.jpg")
                success = cv2.imwrite(tmp, frame_to_save)

                if success:
                    captured_paths.append(tmp)
                    frames_captured += 1
                else:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

            if key == ord("q") or done:
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_paths:
            ok2 = self._ctrl.user_manager.register_face(username, captured_paths)
            for p in captured_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            if ok2:
                QMessageBox.information(self, "Success", f"Face enrolled for '{username}'.")
            else:
                QMessageBox.warning(self, "Failed", "No valid face detected in captured images.")
        self.refresh()

    def _remove_face(self):
        username, ok = QInputDialog.getText(self, "Remove Face", "Username:")
        if not ok or not username:
            return
        if self._ctrl.face_manager.remove_face(username):
            QMessageBox.information(self, "Removed", f"Face data removed for '{username}'.")
        else:
            QMessageBox.warning(self, "Not Found", f"No face data for '{username}'.")
        self.refresh()

    def _change_password(self):
        new_pwd = self._new_pwd_input.text()
        if len(new_pwd) < 8:
            QMessageBox.warning(self, "Too Short", "Password must be at least 8 characters.")
            return
        username = self._ctrl.current_user
        if self._ctrl.user_manager.change_password(username, new_pwd):
            QMessageBox.information(self, "Updated", "Password changed successfully.")
            self._new_pwd_input.clear()
        else:
            QMessageBox.critical(self, "Error", "Could not update password.")


# ─────────────────────────────────────────────────────────────────────────────
# Screen 6 — Auth History
# ─────────────────────────────────────────────────────────────────────────────

class HistoryScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)

        hdr = QHBoxLayout()
        title = QLabel("Auth History")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()
        refresh_btn = QPushButton("🔄  Refresh")
        refresh_btn.clicked.connect(self.refresh)
        hdr.addWidget(refresh_btn)
        root.addLayout(hdr)
        root.addWidget(_hline())

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Timestamp", "User", "Event", "Method"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self._table)

    def refresh(self):
        events = get_auth_events(200)
        self._table.setRowCount(len(events))
        for row, e in enumerate(events):
            self._table.setItem(row, 0, QTableWidgetItem(e.get("timestamp", "")[:16]))
            self._table.setItem(row, 1, QTableWidgetItem(e.get("username") or "—"))
            self._table.setItem(row, 2, QTableWidgetItem(e.get("event_type", "")))
            self._table.setItem(row, 3, QTableWidgetItem(e.get("method") or "—"))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

class Sidebar(QFrame):
    navigate = pyqtSignal(str)

    _PAGES = [
        ("🏠", "Dashboard",  "dashboard"),
        ("📁", "Files",      "files"),
        ("👁️", "Intruders",  "logs"),
        ("📊", "History",    "history"),
        ("⚙️", "Settings",   "settings"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(160)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME_SURFACE};
                border-right: 2px solid #2a2a4a;
            }}
            QPushButton {{
                text-align: left;
                padding: 10px 14px;
                border: none;
                border-radius: 0;
                font-size: 13px;
                background: transparent;
            }}
            QPushButton:hover {{
                background-color: {THEME_ACCENT}33;
                border-left: 3px solid {THEME_ACCENT};
            }}
            QPushButton[active=true] {{
                background-color: {THEME_ACCENT}55;
                border-left: 3px solid {THEME_ACCENT};
                color: #ffffff;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        logo = QLabel("🛡️")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("font-size: 36px; padding: 16px 0;")
        layout.addWidget(logo)

        self._buttons: dict[str, QPushButton] = {}
        for icon, name, key in self._PAGES:
            btn = QPushButton(f"{icon}  {name}")
            btn.clicked.connect(lambda _, k=key: self.navigate.emit(k))
            self._buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch()

    def set_active(self, key: str):
        for k, btn in self._buttons.items():
            btn.setProperty("active", str(k == key).lower())
            btn.style().unpolish(btn)
            btn.style().polish(btn)


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._ctrl = get_controller()
        self.setWindowTitle(APP_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._check_first_run()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Main content = sidebar + stacked pages
        self._content   = QHBoxLayout()
        self._content.setContentsMargins(0, 0, 0, 0)
        self._content.setSpacing(0)

        # Sidebar (hidden on login screen)
        self._sidebar = Sidebar()
        self._sidebar.navigate.connect(self._show_page)
        self._sidebar.hide()
        self._content.addWidget(self._sidebar)

        # Page stack
        self._stack = QStackedWidget()
        self._content.addWidget(self._stack)

        root.addLayout(self._content)

        # ── Instantiate screens ───────────────────────────────────────────────
        self._login_screen    = LoginScreen(self._ctrl)
        self._dashboard       = DashboardScreen(self._ctrl)
        self._files_screen    = FilesBrowserScreen(self._ctrl)
        self._intruder_screen = IntruderLogsScreen()
        self._history_screen  = HistoryScreen()
        self._settings_screen = SettingsScreen(self._ctrl)

        self._pages = {
            "login":     self._login_screen,
            "dashboard": self._dashboard,
            "files":     self._files_screen,
            "logs":      self._intruder_screen,
            "history":   self._history_screen,
            "settings":  self._settings_screen,
        }
        for screen in self._pages.values():
            self._stack.addWidget(screen)

        # Signals
        self._login_screen.authenticated.connect(self._on_authenticated)
        self._dashboard.navigate.connect(self._show_page)
        self._dashboard.lock_req.connect(self._lock_and_return_to_login)

    # ── First-run check ───────────────────────────────────────────────────────

    def _check_first_run(self):
        if not self._ctrl.has_any_user():
            self._run_setup_wizard()
        else:
            self._show_login()

    def _run_setup_wizard(self):
        QMessageBox.information(
            self, "First Run",
            "Welcome to AI SecureVault!\n\nYou need to create an admin account to proceed."
        )
        while True:
            username, ok = QInputDialog.getText(self, "Setup", "Choose a username:")
            if not ok or not username:
                sys.exit(0)

            password, ok = QInputDialog.getText(
                self, "Setup", "Choose a strong password (min 8 chars):", QLineEdit.Password
            )
            if not ok or not password:
                sys.exit(0)

            if len(password) < 8:
                QMessageBox.warning(self, "Too Short", "Password must be at least 8 characters.")
                continue

            if self._ctrl.setup_first_user(username, password):
                QMessageBox.information(
                    self, "Account Created",
                    f"Account '{username}' created!\n\n"
                    "Tip: Go to Settings → Enrol Face after logging in."
                )
                break

        self._show_login()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _show_login(self):
        self._sidebar.hide()
        self._stack.setCurrentWidget(self._login_screen)
        self._login_screen.start()

    def _show_page(self, key: str):
        if key not in self._pages:
            return
        screen = self._pages[key]
        self._stack.setCurrentWidget(screen)
        self._sidebar.set_active(key)
        # Refresh data-driven screens
        if hasattr(screen, "refresh"):
            screen.refresh()

    def _on_authenticated(self, username: str):
        self._sidebar.show()
        self._show_page("dashboard")

    def _lock_and_return_to_login(self):
        self._ctrl.lock_and_logout()
        self._login_screen._msg_label.setText("")
        self._login_screen._password_input.clear()
        self._show_login()

    # ── Window close ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Ensure vault is locked before the process exits."""
        if self._ctrl.vault.is_unlocked:
            reply = QMessageBox.question(
                self, "Exit",
                "The vault is unlocked. Lock and exit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

        self._login_screen.stop()
        if self._ctrl.is_authenticated:
            self._ctrl.lock_and_logout()

        event.accept()