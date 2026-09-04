# TactileGeo

Milestone 1 implementation of the visual-to-tactile geometry conversion pipeline.

## Backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

## Braille runtime

Milestone 2 uses the official Liblouis `louis` Python bindings with the `en-ueb-g2.ctb` table. The production Docker image installs Liblouis and its table files. Local Windows development also needs a native Liblouis installation plus its official Python bindings; the unrelated PyPI package named `louis` is not a Liblouis binding.
