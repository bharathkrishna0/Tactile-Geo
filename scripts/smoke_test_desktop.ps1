<#
.SYNOPSIS
  Installs release\TactileGeo-Setup.exe silently and checks the installed app.

.DESCRIPTION
  Runs with Python, Node.js and Rust removed from PATH, so the installed app
  must be self-contained. Checks: files installed, packaged backend health,
  a real upload -> process -> tactile SVG with Braille, Model B unavailable
  without a key, the desktop shell starting and stopping its backend, uninstall.
#>
param(
  [string]$Installer = (Join-Path $PSScriptRoot "..\release\TactileGeo-Setup.exe"),
  [string]$SampleImage = (Join-Path $PSScriptRoot "..\backend\app\demo_samples\labelled_triangle_worksheet.png")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }
function Check($ok, $message) { if (-not $ok) { throw "FAILED: $message" }; Write-Host "ok  $message" }

$InstallDir = Join-Path $env:LOCALAPPDATA "TactileGeo"
$DataDir = Join-Path $env:LOCALAPPDATA "org.tactilegeo.desktop"
$Installer = (Resolve-Path $Installer).Path
$SampleImage = (Resolve-Path $SampleImage).Path

Step "Strip developer tools from PATH"
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0;$env:SystemRoot\System32\Wbem"
# py.exe (the Python launcher) lives in C:\Windows on the runner; it is not checked.
foreach ($tool in "python", "node", "npm", "cargo", "rustc") {
  Check (-not (Get-Command $tool -ErrorAction SilentlyContinue)) "$tool is not on PATH"
}
Remove-Item Env:OPENROUTER_API_KEY, Env:MODEL_B_ENABLED -ErrorAction SilentlyContinue

Step "Silent install"
$p = Start-Process $Installer -ArgumentList "/S" -Wait -PassThru
Check ($p.ExitCode -eq 0) "installer exit code 0 (got $($p.ExitCode))"
$ShellExe = Join-Path $InstallDir "TactileGeo.exe"
$BackendExe = Join-Path $InstallDir "backend\tactilegeo-backend.exe"
Check (Test-Path $ShellExe) "TactileGeo.exe installed at $ShellExe"
Check (Test-Path $BackendExe) "packaged backend installed"
Check (Test-Path (Join-Path $InstallDir "backend\_internal\liblouis\tables\en-ueb-g2.ctb")) "Liblouis UEB table installed"
Check (Test-Path (Join-Path $InstallDir "backend\_internal\liblouis.dll")) "liblouis.dll installed"
Check (Test-Path (Join-Path $InstallDir "backend\_internal\ocr-models\craft_mlt_25k.pth")) "OCR detector model installed"
Check (Test-Path (Join-Path $InstallDir "backend\_internal\ocr-models\english_g2.pth")) "OCR recogniser model installed"
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$shortcut = Get-ChildItem -Recurse $startMenu -Filter "TactileGeo.lnk" -ErrorAction SilentlyContinue | Select-Object -First 1
Check ($shortcut) "Start Menu shortcut created ($($shortcut.FullName))"
$installedMB = (Get-ChildItem -Recurse -File $InstallDir | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Installed size: {0:N0} MB" -f $installedMB)

Step "Installed backend: health, upload, process"
$port = 18765
$smokeData = Join-Path $PSScriptRoot "..\build\smoke-data"
New-Item -ItemType Directory -Force (Join-Path $smokeData "logs") | Out-Null
$backend = Start-Process $BackendExe -ArgumentList @("--port", $port, "--data-dir", "`"$smokeData`"") -PassThru `
  -RedirectStandardOutput (Join-Path $smokeData "logs\stdout.txt") -RedirectStandardError (Join-Path $smokeData "logs\stderr.txt")
try {
  $base = "http://127.0.0.1:$port"
  $healthy = $false
  $started = Get-Date
  while (((Get-Date) - $started).TotalSeconds -lt 180) {
    try { if ((Invoke-RestMethod "$base/api/health" -TimeoutSec 5).status -eq "ok") { $healthy = $true; break } } catch { Start-Sleep -Seconds 1 }
  }
  Check $healthy ("health endpoint answered after {0:N0} s" -f ((Get-Date) - $started).TotalSeconds)
  Check (-not (Get-NetTCPConnection -LocalPort $port -State Listen | Where-Object { $_.LocalAddress -ne "127.0.0.1" })) "backend listens on 127.0.0.1 only"

  # -Form would send application/octet-stream; the API accepts only PNG/JPEG types.
  $session = curl.exe -sS --fail-with-body -F "image=@$SampleImage;type=image/png" "$base/api/sessions" | ConvertFrom-Json
  Check ($session.session_id) "upload created session $($session.session_id)"
  $t = Get-Date
  $result = Invoke-RestMethod "$base/api/sessions/$($session.session_id)/process" -Method Post -TimeoutSec 300
  $elements = $result.semantic_geometry.element_count
  # StrictMode: elements without Braille have no 'braille' key at all.
  $braille = @($result.semantic_geometry.elements | Where-Object { $_.geometry.PSObject.Properties['braille'] -and $_.geometry.braille }).Count
  Write-Host ("Processed in {0:N1} s: {1} elements, {2} Braille labels, readiness {3}" -f ((Get-Date) - $t).TotalSeconds, $elements, $braille, $result.qa_report.score_0_100)
  Check ($elements -gt 0) "Model A found elements"
  Check ($result.tactile_svg -match "<svg") "tactile SVG produced"
  Check ($braille -gt 0) "OCR + Liblouis produced Braille labels"
  Check ($result.tactile_svg -match 'class="braille"') "Braille dots embossed in the SVG"
  Check (Test-Path (Join-Path $smokeData "uploads")) "uploads written to the per-user data dir"

  $modelB = Invoke-RestMethod "$base/api/model-b/status"
  Check (-not $modelB.available) "Model B reports unavailable without a key ($($modelB.reason))"
} finally {
  if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
}

Step "Desktop shell starts and stops its own backend"
$before = @(Get-Process tactilegeo-backend -ErrorAction SilentlyContinue).Count
$shell = Start-Process $ShellExe -PassThru
$child = $null
for ($i = 0; $i -lt 60 -and -not $child; $i++) {
  Start-Sleep -Seconds 1
  $child = Get-CimInstance Win32_Process -Filter "Name = 'tactilegeo-backend.exe'" | Where-Object { $_.ParentProcessId -eq $shell.Id } | Select-Object -First 1
}
Check ($child) "shell started backend process $($child.ProcessId)"
$conn = $null
for ($i = 0; $i -lt 180 -and -not $conn; $i++) {
  Start-Sleep -Seconds 1
  $conn = Get-NetTCPConnection -OwningProcess $child.ProcessId -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}
Check ($conn -and $conn.LocalAddress -eq "127.0.0.1") "shell backend listening on 127.0.0.1:$($conn.LocalPort)"
Check ((Invoke-RestMethod "http://127.0.0.1:$($conn.LocalPort)/api/health").status -eq "ok") "shell backend healthy"
Check (Test-Path (Join-Path $DataDir "logs\backend.log")) "logs written to $DataDir\logs"
Stop-Process -Id $shell.Id -Force
Start-Sleep -Seconds 5
Check (-not (Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue)) "backend exited when the shell was killed (Job Object)"
Check (@(Get-Process tactilegeo-backend -ErrorAction SilentlyContinue).Count -le $before) "no orphaned backend processes"

Step "Uninstall"
$uninstaller = Join-Path $InstallDir "uninstall.exe"
Check (Test-Path $uninstaller) "uninstaller present"
$p = Start-Process $uninstaller -ArgumentList "/S" -Wait -PassThru
Start-Sleep -Seconds 5
Check (-not (Test-Path $ShellExe)) "app removed by uninstaller"
Check (-not (Test-Path $shortcut.FullName)) "Start Menu shortcut removed"
Write-Host "`nSmoke test passed."
