# Build dist\BlotProfiler\BlotProfiler.exe on Windows and zip it.
#   powershell -ExecutionPolicy Bypass -File build_windows.ps1
# Uses $env:PYTHON if set, otherwise "py -3.12".
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path .venv-win)) {
    if ($env:PYTHON) { & $env:PYTHON -m venv .venv-win } else { py -3.12 -m venv .venv-win }
    if ($LASTEXITCODE -ne 0) { throw 'could not create .venv-win' }
}
$py = '.venv-win\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt pyinstaller pytest
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

& $py -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw 'tests failed' }

& $py -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\blot_profiler_win.spec
if ($LASTEXITCODE -ne 0) { throw 'pyinstaller failed' }

# smoke test: the windowed exe must still be running after 20 s with a TIF open
$exe = 'dist\BlotProfiler\BlotProfiler.exe'
$p = Start-Process -FilePath $exe -ArgumentList 'tests\data\example_blot.tif' -PassThru
Start-Sleep -Seconds 20
if ($p.HasExited) { throw "BlotProfiler.exe exited early (code $($p.ExitCode))" }
Stop-Process -Id $p.Id -Force
Write-Host 'smoke test passed: BlotProfiler.exe started and opened a TIF'

$ver = ([regex]::Match((Get-Content blot_profiler\__init__.py -Raw), '__version__ = "(.+)"')).Groups[1].Value
$name = "BlotProfiler-$ver-win64"
Compress-Archive -Force -Path dist\BlotProfiler -DestinationPath "dist\$name.zip"
if ($env:GITHUB_ENV) { "BUILD_NAME=$name" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8 }
Write-Host "Built dist\$name.zip"
