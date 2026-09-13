"""Exercise native stderr + exit handling under Windows PowerShell 5."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'win32' or not shutil.which('powershell'), reason='Windows PowerShell required')
def test_terminal_progress_and_crash_are_logged(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'start.ps1'
    shutil.copyfile(script, tmp_path/'start.ps1')
    target = tmp_path/'.venv'/'Scripts'/'python.exe'
    target.parent.mkdir(parents=True)
    # A native executable imitates Python's import preflight and failing compile.
    builder = tmp_path/'build.ps1'
    builder.write_text('''$ErrorActionPreference = 'Stop'
$source = @'
using System;
public class FakePython {
  public static int Main(string[] args) {
    if (Array.IndexOf(args, "-c") >= 0) return 0;
    Console.Error.WriteLine("Compiler 3/5: Fortschritt");
    Console.Error.WriteLine("MemoryError: regression simulation");
    return 7;
  }
}
'@
Add-Type -TypeDefinition $source -OutputAssembly ($PSScriptRoot + '/.venv/Scripts/python.exe') -OutputType ConsoleApplication
''', encoding='utf-8')
    subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-File', str(builder)], check=True, capture_output=True, timeout=30)
    process = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-File', str(tmp_path/'start.ps1'), '-NoBrowser'],
                             stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    assert process.returncode == 7
    assert b'Compiler 3/5: Fortschritt' in process.stdout
    log = next((tmp_path/'runtime').glob('start-*.log')).read_text(encoding='utf-8-sig')
    assert 'Compiler 3/5: Fortschritt' in log
    assert 'MemoryError: regression simulation' in log
