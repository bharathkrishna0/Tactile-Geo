# Windows desktop app — implementation plan

Status: **implemented** — see `README.md` in this folder for what was built and how it is verified. This file is the original plan, kept for reference.

## 1. Audit findings (current `main`, 178a6e5)

| Question | Finding | Consequence for packaging |
| --- | --- | --- |
| How does the frontend call FastAPI? | Every call is `${API_BASE}/api/...` (`frontend/src/api/client.ts`, `frontend/src/lib/modelBApi.ts`). `API_BASE` is `VITE_API_BASE_URL` or `''`. | One shared client already exists. The desktop build only needs `API_BASE` to come from the shell at runtime. |
| Does it depend on the Vite proxy? | Yes when `API_BASE` is `''` (dev). Not otherwise. | The packaged app sets an absolute `http://127.0.0.1:<port>`. `npm run dev` is unchanged. |
| Backend dependencies | FastAPI, Uvicorn, python-multipart, OpenCV headless, NumPy, EasyOCR → PyTorch/torchvision, psycopg (only with `DATABASE_URL`). | PyTorch dominates size. Windows PyPI wheels are already CPU-only. |
| Runtime assets | EasyOCR `craft_mlt_25k.pth` (83 MB) and `english_g2.pth` (15 MB), downloaded to `~/.EasyOCR` on first use. `liblouis.dll` + tables are vendored in `backend/.native/win64` (Liblouis 3.38). | Bundle both OCR models (no first-run download, works offline). Bundle `liblouis.dll` + `share/liblouis/tables`. |
| Liblouis Python binding on Windows | Not installed by any current Windows step. The repo has only the template `python/louis/__init__.py.in` (placeholder `###LIBLOUIS_SONAME###`). | Build step generates `louis/__init__.py` from that template with `liblouis.dll`. This is the official binding, not the unrelated PyPI `louis`. |
| Working-directory assumptions | `UPLOAD_DIRECTORY` defaults to relative `uploads`. Liblouis tables are found relative to `backend/` or via `LOUIS_TABLEPATH`. | Desktop entrypoint sets absolute per-user paths and `LOUIS_TABLEPATH` before importing the app. |
| Uploads, temp files, logs | Uploads in `UPLOAD_DIRECTORY`, deleted after `SESSION_TTL_SECONDS` (2 h). No log files today (stdout only). | `%LOCALAPPDATA%\TactileGeo\uploads` and `...\logs`. Install dir stays read-only. |
| Sessions and jobs | In-process memory store unless `DATABASE_URL` is set. Model B jobs run in-process. | Desktop uses the memory store: no database, nothing leaves the laptop. Sessions end when the app closes. |
| Internet needed? | Only Model B (OpenRouter). Model A, OCR, Braille, QA, SVG and Template Demo are local once OCR models are present. | Desktop is offline-capable except Model B. Model B stays **off** unless the user sets `MODEL_B_ENABLED` and `OPENROUTER_API_KEY` as Windows user environment variables. |
| SVG export | Blob + `<a download>` in `App.tsx` `exportSvg()`. | In desktop mode, use a native Save dialog instead (Tauri dialog + fs plugins). Browser path unchanged. |
| Text-first OCR / Tesseract | **PR #13's code is not on `main`.** It was merged into the PR #12 branch after #12 had already merged. | Package `main` as it is: EasyOCR only, no Tesseract. Restoring #13 is a separate PR. |

## 2. Architecture (Tauri 2 + existing React build + PyInstaller backend)

```text
TactileGeo.exe (Tauri 2, Rust, WebView2)
  ├─ picks a free 127.0.0.1 port (bind :0)
  ├─ starts resources\backend\tactilegeo-backend.exe --port N  (CREATE_NO_WINDOW, inside a Job Object: killed if the shell exits or crashes)
  ├─ injects window.__TACTILEGEO_DESKTOP__ = { apiBase: "http://127.0.0.1:N", logsDir } before the page loads
  └─ loads the existing frontend/dist (same React app, same Template Demo assets)

tactilegeo-backend.exe (PyInstaller --onedir)
  └─ backend/desktop_main.py: set per-user paths, LOUIS_TABLEPATH, EASYOCR model dir, CORS for the Tauri origin → uvicorn app.main:app on 127.0.0.1 only
```

Why these choices:
- **Tauri 2** (as the brief prefers): small shell, uses Windows' built-in WebView2, NSIS installer with Start Menu shortcut and uninstaller included.
- **PyInstaller `--onedir`, launched from Rust rather than as a Tauri `externalBin` sidecar.** `externalBin` takes a single executable, which would mean `--onefile`. `--onefile` unpacks the whole PyTorch tree (hundreds of MB) to `%TEMP%` on every launch. `--onedir` starts faster and is easier to debug.
- **Job Object** guarantees no orphaned backend, and only the process the app started is ever terminated.

## 3. Changes (additive; pipeline untouched)

Frontend (small):
- `lib/apiBase.ts`: `API_BASE = window.__TACTILEGEO_DESKTOP__?.apiBase ?? VITE_API_BASE_URL ?? ''`.
- Desktop only: a "Starting local engine…" state that polls `/api/health` before enabling upload, with an error and log location on timeout. `exportSvg()` uses a Save dialog. The Download Desktop App button is hidden (needs PR #15).

Backend (small):
- New `backend/desktop_main.py` entrypoint. Model A, Model B, fusion, QA and SVG code unchanged.
- `services/ocr.py`: honour an `EASYOCR_MODEL_DIR` env var (`model_storage_directory=..., download_enabled=False`). Unset = current behaviour.
- Rotating log file when `TACTILEGEO_LOG_DIR` is set. No secrets or document text logged.

New:
- `desktop/src-tauri/` (Rust shell, `tauri.conf.json`, icons, Help → Open Logs Folder, About).
- `backend/desktop-build/tactilegeo-backend.spec` (PyInstaller) and `make_louis_binding.py`.
- `scripts/build_desktop.ps1`: checks prerequisites → frontend build → backend PyInstaller → assemble resources → `tauri build` (NSIS) → `release\TactileGeo-Setup.exe` + `SHA256SUMS.txt`, printing the path and size.
- `.github/workflows/desktop.yml` on `windows-latest`. Free: GitHub Actions on standard runners is free for public repos.
  - Runs on manual dispatch, or on a `v*` tag push.
  - Builds the installer and uploads it as a workflow artifact.
  - On a tag, creates a **draft** GitHub Release with `TactileGeo-Setup.exe` and `SHA256SUMS.txt`. You publish the draft, then put its URL in `frontend/src/desktopRelease.ts`.
- `docs/desktop/README.md` covering everything the brief's section 16 asks for.

## 4. Verification

| Check | Where | Automated? |
| --- | --- | --- |
| Existing backend + frontend tests, typecheck, build | Linux (here) | Yes |
| PyInstaller backend starts, `/api/health` ok, processes a sample image with Braille output, exports SVG | Windows CI | Yes |
| Silent install (`/S`) → run backend from install dir with no Python on PATH → uninstall leaves no install dir | Windows CI | Yes |
| Port conflict (port taken) → picks another; missing/invalid Model B key → Model A still works | Windows CI (backend level) | Yes |
| App window opens, Template Demo offline, Save dialog, close kills backend, Start Menu shortcut | Your Windows laptop | **No — needs you** (checklist in the README) |
| Clean Windows machine without dev tools | Your laptop or a VM | **No — needs you** |

I can't run a Windows GUI from this Linux machine, so the window-level checks stay unverified until you run them.

## 5. Risks

- **Size (estimate, unmeasured):** about 1–1.5 GB installed, roughly 400–800 MB installer. Will be measured and reported. GitHub caps release files at 2 GiB.
- **Unsigned installer:** Windows SmartScreen will warn "Windows protected your PC" (More info → Run anyway). Code signing costs money, though free signing for open-source projects (e.g. SignPath Foundation) could be applied for later.
- **WebView2:** present on Windows 11 and current Windows 10. If it's missing, the installer downloads it, which needs internet once.
- **PyInstaller + PyTorch/EasyOCR hidden imports:** the likeliest build failures. Expect several Windows CI rounds (~20–30 min each).
- **First OCR call** loads PyTorch and models, so expect a slow first image (seconds to tens of seconds). The UI will show progress instead of hanging.
- **Model B credentials:** stored as Windows user environment variables, documented. A Credential Manager UI is out of scope for v1.
