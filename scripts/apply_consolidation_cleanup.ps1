# Apply only the verified, hash-bound local dataset cleanup plan. No backups.
$ErrorActionPreference = 'Stop'
$freqaiRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$freqaiPlan = Get-Content -LiteralPath (Join-Path $freqaiRoot 'results\memory_cleanup_plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json

function Resolve-FreqaiPath([string]$Relative) {
    $resolved = [IO.Path]::GetFullPath((Join-Path $freqaiRoot $Relative))
    if (-not $resolved.StartsWith($freqaiRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Ziel liegt außerhalb des Projekts: $Relative"
    }
    $ancestor = $resolved
    while ($ancestor -ne $freqaiRoot) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Verknüpfungen werden nicht aufgeräumt: $ancestor"
            }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    return $resolved
}

$freqaiDatabase = Resolve-FreqaiPath $freqaiPlan.central_database
$freqaiStateJson = & (Join-Path $freqaiRoot '.venv\Scripts\python.exe') -c 'import json,sqlite3,sys; from pathlib import Path; c=sqlite3.connect(Path(sys.argv[1]).as_uri()+"?mode=ro",uri=True); print(json.dumps({"revision":c.execute("SELECT value FROM metadata WHERE key=''revision''").fetchone()[0],"documents":c.execute("SELECT COUNT(*) FROM documents").fetchone()[0],"archive":c.execute("SELECT COUNT(*) FROM document_archive").fetchone()[0]}))' $freqaiDatabase
if ($LASTEXITCODE -ne 0) { throw 'Zentrale Memory konnte nicht geprüft werden.' }
$freqaiState = $freqaiStateJson | ConvertFrom-Json
if ($freqaiState.revision -ne $freqaiPlan.revision -or $freqaiState.documents -ne 569210 -or
    $freqaiState.archive -ne 0 -or $freqaiPlan.articles_verified -ne 569062) {
    throw 'Memory stimmt nicht mit dem geprüften Import überein.'
}

# Preflight every target before the first filesystem change.
foreach ($entry in $freqaiPlan.files) {
    if ($entry.path -notmatch '^(memory/(backups|\.compiler-cache|imports|information|language|fixtures|sources)/|dataset/|results/|runtime/|unsloth-tmp/|recs\.pkl$|tests/fixtures/benchmark\.json$)') {
        throw "Nicht zugelassener Löschbereich: $($entry.path)"
    }
    $target = Resolve-FreqaiPath $entry.path
    if (Test-Path -LiteralPath $target) {
        if ((Get-Item -LiteralPath $target).Length -ne $entry.bytes -or
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $entry.sha256) {
            throw "Datei seit der Prüfung geändert: $target"
        }
    }
}
foreach ($entry in $freqaiPlan.moves) {
    if ($entry.path -notlike 'memory/sources/*' -or $entry.destination -notlike 'docs/data_sources/*') {
        throw 'Nicht zugelassene Dokumentationsverschiebung.'
    }
    $source = Resolve-FreqaiPath $entry.path
    $destination = Resolve-FreqaiPath $entry.destination
    if (Test-Path -LiteralPath $source) {
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $entry.sha256 -or
            (Test-Path -LiteralPath $destination)) { throw "Dokumentationskonflikt: $source" }
    } elseif (-not (Test-Path -LiteralPath $destination) -or
        (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $entry.sha256) {
        throw "Dokumentation fehlt: $source"
    }
}

foreach ($entry in $freqaiPlan.moves) {
    $source = Resolve-FreqaiPath $entry.path
    $destination = Resolve-FreqaiPath $entry.destination
    if (Test-Path -LiteralPath $source) {
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
        Move-Item -LiteralPath $source -Destination $destination
    }
}
$freqaiDeleted = 0
$freqaiBytes = [long]0
foreach ($entry in $freqaiPlan.files) {
    $target = Resolve-FreqaiPath $entry.path
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Force
        $freqaiDeleted++
        $freqaiBytes += $entry.bytes
    }
}
foreach ($relative in $freqaiPlan.empty_directories) {
    $directory = Resolve-FreqaiPath $relative
    if ((Test-Path -LiteralPath $directory -PathType Container) -and
        @(Get-ChildItem -LiteralPath $directory -Force).Count -eq 0) {
        Remove-Item -LiteralPath $directory -Force
    }
}
$freqaiRemaining = @($freqaiPlan.files | Where-Object { Test-Path -LiteralPath (Resolve-FreqaiPath $_.path) })
if ($freqaiRemaining.Count) { throw 'Es sind noch vorgemerkte Datensatzdateien vorhanden.' }
@{completed_utc=[DateTime]::UtcNow.ToString('o'); deleted_files=$freqaiDeleted; deleted_bytes=$freqaiBytes;
  planned_files=$freqaiPlan.delete_count; remaining_files=0; central_revision=$freqaiState.revision;
  central_documents=$freqaiState.documents; metadata_moves=$freqaiPlan.moves.Count} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $freqaiRoot 'results\memory_cleanup_result.json') -Encoding UTF8
Write-Output "$freqaiDeleted Dateien gelöscht; $freqaiBytes Bytes freigegeben. Zentrale Memory erhalten."
