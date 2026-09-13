param([int]$Port = 8765, [switch]$NoBrowser)
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
$runtime = Join-Path $PSScriptRoot 'runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$logPath = Join-Path $runtime ('start-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Write-Host "FreqAI startet. Fortschritt und Fehlerprotokoll: $logPath"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONFAULTHANDLER = '1'
$env:PYTHONIOENCODING = 'utf-8'
$previousEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Set-Content -LiteralPath $logPath -Value "FreqAI Start $(Get-Date -Format o)" -Encoding UTF8
try {
    $freqaiArguments = @('-u', '-X', 'faulthandler', '-m', 'freqai', 'serve', '--port', $Port,
                        '--memory', (Join-Path $PSScriptRoot 'memory\memory.sqlite3'))
    if (-not $NoBrowser) { $freqaiArguments += '--open' }
    # Windows PowerShell 5 treats redirected native stderr as ErrorRecord.
    # Progress on stderr is ordinary output, not a reason to abort the pipeline.
    $ErrorActionPreference = 'Continue'
    & $python @freqaiArguments 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
        Write-Host $line
    }
    $ErrorActionPreference = 'Stop'
    $freqaiExit = $LASTEXITCODE
    if ($freqaiExit -ne 0) {
        Add-Content -LiteralPath $logPath -Value "FreqAI Exitcode: $freqaiExit" -Encoding UTF8
        Write-Host "FreqAI wurde mit Exitcode $freqaiExit beendet. Protokoll: $logPath" -ForegroundColor Red
        if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
            Read-Host 'Enter zum Schließen' | Out-Null
        }
        exit $freqaiExit
    }
} finally {
    [Console]::OutputEncoding = $previousEncoding
}
