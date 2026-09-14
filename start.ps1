param([int]$Port = 8765, [switch]$NoBrowser, [int]$RunSeconds = 0)
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
# Kill-on-Close-Job: Alle Kindprozesse des Startskripts (der Server samt seinen
# Rechenarbeitern) werden zuverlaessig beendet, sobald dieser Startprozess endet -
# sei es durch Strg+C, durch einen normalen Exit oder durch das Schliessen des
# Konsolenfensters. Der Job haelt ein Handle; schliesst der Kernel dieses Handle beim
# Prozessende, terminiert Windows zwingend jedes Mitglied des Jobs. Ein eigenes
# CloseHandle im finally waere gefaehrlich, weil der Startprozess selbst im Job sitzt
# und sonst beim normalen Ende sein eigenes Terminate ausloesen wuerde.
$consoleJobCode = @'
using System;
using System.Runtime.InteropServices;
namespace Native
{
    public static class Win32ConsoleJob
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public long Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct IO_COUNTERS
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }
        [DllImport("kernel32.dll")]
        public static extern IntPtr GetCurrentProcess();
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        public static extern IntPtr CreateJobObjectW(IntPtr jobAttributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr jobObjectInfo, uint jobObjectInfoLength);
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        public static IntPtr CreateWithKillOnClose()
        {
            IntPtr job = CreateJobObjectW(IntPtr.Zero, null);
            if (job == IntPtr.Zero)
            {
                return IntPtr.Zero;
            }
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            info.BasicLimitInformation.LimitFlags = 0x00002000;
            int size = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
            IntPtr infoPointer = Marshal.AllocHGlobal(size);
            try
            {
                Marshal.StructureToPtr(info, infoPointer, false);
                if (!SetInformationJobObject(job, 9, infoPointer, (uint)size))
                {
                    CloseHandle(job);
                    return IntPtr.Zero;
                }
            }
            finally
            {
                Marshal.FreeHGlobal(infoPointer);
            }
            return job;
        }
        public static bool AssignCurrentProcess(IntPtr job)
        {
            return AssignProcessToJobObject(job, GetCurrentProcess());
        }
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool CloseHandle(IntPtr handle);
    }
}
'@
Add-Type -TypeDefinition $consoleJobCode
$consoleJobHandle = [Native.Win32ConsoleJob]::CreateWithKillOnClose()
if ($consoleJobHandle -eq [IntPtr]::Zero) {
    Write-Host 'WARNUNG: Kill-on-Close-Job konnte nicht erstellt werden. Der Server wird beim Schliessen des Fensters moeglicherweise nicht automatisch beendet.' -ForegroundColor Yellow
} elseif (-not [Native.Win32ConsoleJob]::AssignCurrentProcess($consoleJobHandle)) {
    Write-Host 'WARNUNG: Kill-on-Close-Job konnte dem Startprozess nicht zugewiesen werden. Der Server wird beim Schliessen des Fensters moeglicherweise nicht automatisch beendet.' -ForegroundColor Yellow
}
try {
    $freqaiArguments = @('-u', '-X', 'faulthandler', '-m', 'freqai', 'serve', '--port', $Port,
                        '--memory', (Join-Path $PSScriptRoot 'memory\memory.sqlite3'))
    if (-not $NoBrowser) { $freqaiArguments += '--open' }
    # Windows PowerShell 5 treats redirected native stderr as ErrorRecord.
    # Progress on stderr is ordinary output, not a reason to abort the pipeline.
    $ErrorActionPreference = 'Continue'
    if ($RunSeconds -gt 0) {
        # Testbetrieb: Der Startprozess terminiert sich selbst nach der Haltezeit.
        # Ueber den Kill-on-Close-Job werden dabei auch der Server und seine
        # Rechenarbeiter beendet, ohne dass der Server ein eigenes Argument kennt.
        $self = [System.Diagnostics.Process]::GetCurrentProcess()
        (Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-WindowStyle', 'Hidden', '-Command', "Start-Sleep -Seconds $RunSeconds; Stop-Process -Id $($self.Id) -Force") -PassThru -WindowStyle Hidden) | Out-Null
    }
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
