# TactileGeo desktop app (Windows)

The existing React frontend and FastAPI backend, packaged as one Windows installer.
Nothing is rewritten: the window shows `frontend/dist`, and the backend is the same
`app.main:app`, frozen with PyInstaller.

```text
TactileGeo.exe  (Tauri 2, desktop/src-tauri)
  ├─ picks a free 127.0.0.1 port
  ├─ starts backend\tactilegeo-backend.exe --port N --data-dir %LOCALAPPDATA%\org.tactilegeo.desktop
  │    (in a kill-on-close Job Object, so it dies with the app, even on a crash)
  ├─ injects window.__TACTILEGEO_DESKTOP__ = { apiBase, logsDir, version, startupError }
  └─ shows frontend/dist; live processing unlocks once /api/health answers

tactilegeo-backend.exe  (PyInstaller one-folder, backend/desktop_main.py)
  ├─ Python 3.11, FastAPI, Uvicorn, OpenCV, NumPy, EasyOCR, CPU PyTorch
  ├─ liblouis.dll 3.38.0 + its tables + the official ctypes binding
  ├─ EasyOCR weights (craft_mlt_25k.pth, english_g2.pth); downloads disabled
  └─ binds 127.0.0.1 only; CORS allows the Tauri origin only
```

| Path | Contents |
| --- | --- |
| `%LOCALAPPDATA%\TactileGeo\` | Installed program (read-only at run time) |
| `%LOCALAPPDATA%\org.tactilegeo.desktop\uploads\` | Session images |
| `%LOCALAPPDATA%\org.tactilegeo.desktop\logs\` | `backend.log` (rotated), `backend-console.log` |

Help → Open logs folder opens the logs. The browser build is unchanged:
`window.__TACTILEGEO_DESKTOP__` is absent there, so `VITE_API_BASE_URL` and the
Vite `/api` proxy work as before, and the Download Desktop App button shows only
in the browser.

## Model B credentials

No key is in the installer, the frontend bundle, or the repository. The teacher
enters their own OpenRouter key under **Model B key**. The shell stores it in
Windows Credential Manager (service `TactileGeo`, user `openrouter-api-key`),
and passes it to the backend process environment at start, with
`MODEL_B_ENABLED=true`. The page never reads the key back. Restart the app after
saving or removing it. Without a key, Model B reports unavailable and Model A
works normally.

## Build

GitHub Actions builds it (`.github/workflows/desktop.yml`). Every PR that touches
the app runs build + smoke test; pushing a tag `vX.Y.Z` that matches
`desktop/src-tauri/tauri.conf.json` also publishes the GitHub Release and checks
the public download URL.

Local build on Windows (needs Python 3.11, Node.js 22, Rust with the MSVC
toolchain, and the Visual Studio C++ build tools; end users need none of these):

```powershell
pwsh scripts\build_desktop.ps1
pwsh scripts\smoke_test_desktop.ps1   # installs, tests and uninstalls
```

The build stops at the first failing step. Before the installer is made, the
frozen backend runs `--self-check`, which translates Braille with the bundled
Liblouis, loads the bundled OCR weights, and runs Model A on a generated
triangle; any failure fails the build.

Development without packaging: `TACTILEGEO_DEV_PYTHON=backend\.venv\Scripts\python.exe`
then `npx tauri dev` in `desktop/` runs `backend/desktop_main.py` from source.

## What the CI smoke test checks

Runs on a fresh Windows runner with Python, Node.js and Rust removed from PATH:
silent install; program files, Liblouis DLL and tables, OCR weights and Start
Menu shortcut present; packaged backend healthy on 127.0.0.1 only; a real upload
→ process → SVG with Braille dots; Model B unavailable without a key; the app
starting its own backend and the backend dying when the app is killed; silent
uninstall.

It does not look at the window itself (CI has no one to look), and the runner
already has WebView2 and the Visual C++ runtime, so it is not a clean machine.

## Clean-machine test plan

Use a Windows 10 or 11 laptop or VM without Python, Node.js, Rust or the
repository (check with `where python node cargo` in a terminal: all should fail).

1. Download `TactileGeo-Setup.exe` from the release page and check its SHA-256
   (`Get-FileHash .\TactileGeo-Setup.exe`) against `SHA256SUMS.txt`.
2. Run it. If SmartScreen warns (the installer is unsigned), choose More info →
   Run anyway. Finish the installer.
3. Start TactileGeo from the Start Menu. Expect "Starting the local TactileGeo
   engine…", then Choose image becomes clickable (first start can take up to a minute).
4. Disconnect from the internet. Upload a worksheet PNG/JPG, click Create
   tactile preview, and check shapes, labels and Braille appear.
5. Export print-ready SVG: a Save dialog opens; save it and open the file in a
   browser.
6. Click Template (Demo), step through all 7 stages, and return to live mode.
7. Model B without a key: it shows as unavailable; Model A still works.
8. Reconnect, open Model B key, save an invalid key, restart the app, and run
   Model B: expect an error message, not a crash. Remove the key.
9. Help → Open logs folder: `backend.log` exists.
10. Close the app; in Task Manager, no `tactilegeo-backend.exe` remains.
11. Port conflict: start the app twice; both should work (each picks a free port).
12. Run the installer again over the existing install (reinstall), then
    uninstall from Settings → Apps. The program folder and shortcut are gone.

Report the Windows version, the result of each step, and `backend.log` for any failure.

## Remaining requirements and internet use

- **Windows 10/11 64-bit**, about 2.5 GB of free RAM while processing.
- **WebView2:** preinstalled on Windows 11 and updated Windows 10. If it is
  missing, the installer's embedded bootstrapper downloads it (internet needed
  once). The offline WebView2 installer (~130 MB more) is not bundled.
- **Visual C++ runtime:** PyInstaller bundles the Python and PyTorch DLLs it
  finds, but this has only been checked on GitHub's runner, which already has
  the runtime installed. The clean-machine test above is the real check.
- **Model B:** internet and the teacher's own OpenRouter key.
- **Tesseract:** the optional second OCR engine is not bundled; EasyOCR alone is used.
- **Not code-signed:** SmartScreen warns on first run.
