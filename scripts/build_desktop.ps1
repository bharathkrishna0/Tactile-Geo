<#
.SYNOPSIS
  Builds release\TactileGeo-Setup.exe (Tauri 2 shell + PyInstaller backend).

.DESCRIPTION
  Needs Python 3.11, Node.js 20+, and Rust (MSVC) on the build machine only.
  Every step stops the build on failure; required runtime assets are verified
  by the packaged backend's --self-check before the installer is made.
  Run from the repository root:  pwsh scripts\build_desktop.ps1
#>
param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Build = Join-Path $Root "build\desktop"
$Stage = Join-Path $Build "stage"
$Venv = Join-Path $Build "venv"
$BackendDist = Join-Path $Build "backend"
$Release = Join-Path $Root "release"

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }
function Require($path, $what) { if (-not (Test-Path $path)) { throw "Missing ${what}: $path" } }

Step "Toolchain"
& $Python --version
node --version
cargo --version
$pyVersion = & $Python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ($pyVersion -ne "3.11") { throw "Python 3.11 is required (found $pyVersion)" }

Remove-Item -Recurse -Force $Build, $Release -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Build, $Release | Out-Null

Step "Frontend production build"
Push-Location (Join-Path $Root "frontend")
npm ci --no-audit --no-fund
npm run build
Pop-Location
Require (Join-Path $Root "frontend\dist\index.html") "frontend build"
Require (Join-Path $Root "frontend\dist\demo\tactile-template\metadata.json") "Template Demo assets"

Step "Backend build environment"
& $Python -m venv $Venv
$VenvPython = Join-Path $Venv "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "backend\desktop\requirements-desktop.txt")

Step "Stage Liblouis and OCR models"
& $VenvPython (Join-Path $Root "backend\desktop\stage_resources.py") --out $Stage

Step "PyInstaller (one-folder backend)"
$env:TACTILEGEO_STAGE = $Stage
& $VenvPython -m PyInstaller --noconfirm --clean `
  --distpath $BackendDist --workpath (Join-Path $Build "pyinstaller-work") `
  (Join-Path $Root "backend\desktop\tactilegeo-backend.spec")
$BackendExe = Join-Path $BackendDist "tactilegeo-backend\tactilegeo-backend.exe"
Require $BackendExe "packaged backend"

Step "Packaged backend self-check (Liblouis, OCR models, Model A)"
$Report = Join-Path $Build "self-check.json"
$CheckData = Join-Path $Build "self-check-data"
$proc = Start-Process -FilePath $BackendExe -ArgumentList @("--data-dir", "`"$CheckData`"", "--self-check", "`"$Report`"") -Wait -PassThru -NoNewWindow
Get-Content $Report -ErrorAction SilentlyContinue | Write-Host
if ($proc.ExitCode -ne 0) {
  Get-Content (Join-Path $CheckData "logs\backend.log") -ErrorAction SilentlyContinue | Write-Host
  throw "Packaged backend self-check failed (exit $($proc.ExitCode))"
}

Step "Tauri installer"
Push-Location (Join-Path $Root "desktop")
npm ci --no-audit --no-fund
npx tauri build --bundles nsis
Pop-Location
$Nsis = Get-ChildItem (Join-Path $Root "desktop\src-tauri\target\release\bundle\nsis") -Filter "*-setup.exe" | Select-Object -First 1
if (-not $Nsis) { throw "NSIS installer was not produced" }

Step "Release folder"
$Setup = Join-Path $Release "TactileGeo-Setup.exe"
Copy-Item $Nsis.FullName $Setup
$hash = (Get-FileHash $Setup -Algorithm SHA256).Hash.ToLower()
"$hash  TactileGeo-Setup.exe" | Set-Content -Encoding ascii (Join-Path $Release "SHA256SUMS.txt")
$size = (Get-Item $Setup).Length
$backendSize = (Get-ChildItem -Recurse -File (Join-Path $BackendDist "tactilegeo-backend") | Measure-Object Length -Sum).Sum
Write-Host ("Installer: {0} ({1:N0} bytes, {2:N1} MB)" -f $Setup, $size, ($size / 1MB))
Write-Host ("Packaged backend folder: {0:N1} MB" -f ($backendSize / 1MB))
if ($size -ge 2GB) { throw "Installer exceeds the 2 GiB GitHub Release asset limit" }
