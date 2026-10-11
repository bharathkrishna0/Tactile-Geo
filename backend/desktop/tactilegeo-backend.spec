# PyInstaller spec for the packaged desktop backend (one-folder build).
# Run via scripts/build_desktop.ps1, which stages assets into TACTILEGEO_STAGE first.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

backend = Path(SPECPATH).parent
stage = Path(os.environ["TACTILEGEO_STAGE"]).resolve()

datas = [
    (str(backend / "app" / "demo_samples"), "app/demo_samples"),
    (str(stage / "liblouis" / "tables"), "liblouis/tables"),
    (str(stage / "ocr-models"), "ocr-models"),
    *collect_data_files("easyocr"),
]
binaries = [(str(stage / "liblouis" / "liblouis.dll"), ".")]
hiddenimports = [
    *collect_submodules("app"),
    *collect_submodules("uvicorn"),
    "louis",
]

a = Analysis(
    [str(backend / "desktop_main.py")],
    pathex=[str(backend), str(stage / "louis-binding")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="tactilegeo-backend",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="tactilegeo-backend")
