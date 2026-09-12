param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python-Umgebung konnte nicht erstellt werden.' }
}
& $python -c "import importlib.util, sys; sys.exit(not all(importlib.util.find_spec(name) for name in ('numpy', 'scipy', 'threadpoolctl')))"
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install -e '.[experiments]'
    if ($LASTEXITCODE -ne 0) { throw 'Module konnten nicht installiert werden.' }
}
& $python -m freqai serve --port $Port --open
