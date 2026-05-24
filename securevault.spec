# securevault.spec — PyInstaller build spec
# Build with:  pyinstaller securevault.spec

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect dlib / face_recognition model data
datas = []
datas += collect_data_files("face_recognition_models")

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas + [
        ("data", "data"),           # bundle the data folder
        ("config.py", "."),
    ],
    hiddenimports=[
        "face_recognition",
        "face_recognition_models",
        "dlib",
        "cv2",
        "cryptography",
        "cryptography.hazmat.primitives.kdf.pbkdf2",
        "bcrypt",
        "scipy.spatial",
        "scipy.spatial.distance",
        "PyQt5.sip",
    ] + collect_submodules("sklearn"),
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SecureVault",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # hide console window in production
    icon="icon.ico",  # replace with your icon path
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SecureVault",
)
