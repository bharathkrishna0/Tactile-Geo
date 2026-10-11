# PyInstaller spec for the packaged desktop backend (one-folder build).
# Run via scripts/build_desktop.ps1, which stages assets into TACTILEGEO_STAGE first.
import importlib.util
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
# torchvision loads its C++ ops (nms, used by EasyOCR) with torch.ops.load_library
# from its package folder; the hook only knows the old "_C" name, not "_C_stable".
torchvision_dir = Path(importlib.util.find_spec("torchvision").origin).parent
torchvision_ops = sorted(torchvision_dir.glob("*.pyd"))
if not any(p.stem.startswith("_C") for p in torchvision_ops):
    raise SystemExit(f"torchvision C++ ops not found in {torchvision_dir}")
binaries = [
    (str(stage / "liblouis" / "liblouis.dll"), "."),
    *[(str(p), "torchvision") for p in [*torchvision_ops, *torchvision_dir.glob("*.dll")]],
]
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
