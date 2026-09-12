$ErrorActionPreference = 'Stop'
$metadataPath = Join-Path $PSScriptRoot 'runtime\server-process.json'
if (-not (Test-Path -LiteralPath $metadataPath)) {
    Write-Output 'Kein aufgezeichneter Hintergrundprozess. Einen Konsolenstart mit Strg+C beenden.'
    exit 0
}
$metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
$serverProcessId = [int]$metadata.pid
$serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $serverProcessId"
if (-not $serverProcess) {
    Write-Output 'Der aufgezeichnete FreqAI-Prozess ist bereits beendet.'
    exit 0
}
if ($metadata.creationTime -and
    $serverProcess.CreationDate.ToUniversalTime() -ne ([DateTimeOffset]$metadata.creationTime).UtcDateTime) {
    throw 'Die Prozess-ID wurde inzwischen neu vergeben. Es wird kein Prozess beendet.'
}
$expectedMemory = Join-Path $PSScriptRoot 'memory\memory.sqlite3'
if (-not $serverProcess.CommandLine.Contains('-m freqai serve') -or
    -not $serverProcess.CommandLine.Contains($expectedMemory)) {
    throw 'Die Prozess-ID gehört nicht mehr zu diesem FreqAI-Server. Es wird kein Prozess beendet.'
}
# Spawn-Worker müssen vor ihrem Elternprozess beendet werden. Nach dessen Ende
# wären ihre Prozess-IDs nicht mehr sicher diesem Server zuzuordnen.
$freqaiProcesses = @(Get-CimInstance Win32_Process)
$freqaiTree = [System.Collections.Generic.List[uint32]]::new()
$freqaiTree.Add([uint32]$serverProcessId)
for ($freqaiIndex = 0; $freqaiIndex -lt $freqaiTree.Count; $freqaiIndex++) {
    foreach ($freqaiChild in $freqaiProcesses | Where-Object ParentProcessId -eq $freqaiTree[$freqaiIndex]) {
        $freqaiParentId = $freqaiTree[$freqaiIndex]
        $freqaiCommand = [string]$freqaiChild.CommandLine
        $freqaiSpawn = $freqaiCommand.Contains('multiprocessing.spawn') -and
            $freqaiCommand.Contains('--multiprocessing-fork') -and
            $freqaiCommand -match ('parent_pid\s*=\s*' + $freqaiParentId + '(?:\s*,|\s*\))')
        if ($freqaiChild.Name -eq 'python.exe' -and ($freqaiSpawn -or
            $freqaiCommand.Contains((Join-Path $PSScriptRoot '.venv\Scripts\python.exe')))) {
            $freqaiTree.Add([uint32]$freqaiChild.ProcessId)
        }
    }
}
$freqaiTree.Reverse()
foreach ($freqaiProcessId in $freqaiTree) {
    $freqaiOriginal = $freqaiProcesses | Where-Object ProcessId -eq $freqaiProcessId
    $freqaiCurrent = Get-CimInstance Win32_Process -Filter "ProcessId = $freqaiProcessId"
    if ($freqaiCurrent -and $freqaiCurrent.CreationDate -eq $freqaiOriginal.CreationDate) {
        Stop-Process -Id $freqaiProcessId -ErrorAction SilentlyContinue
    }
}
Write-Output "FreqAI-Hintergrundprozess $serverProcessId beendet. Gespeicherte Daten bleiben erhalten."
