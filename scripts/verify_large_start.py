"""Run the real start.ps1 with a process-tree RAM watchdog and startup receipt.

The server remains running on success; only this launch's process tree is stopped
on timeout, memory-limit violation or error. Run again on another free port for
a warm-cache measurement after stopping the first instance.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--name', default='large-start')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--max-gib', type=float, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = root/'runtime'
    runtime.mkdir(exist_ok=True)
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('Testport ist bereits belegt; vorhandenen Server zuerst beenden.')
    started = time.monotonic()
    peak = 0
    receipt = {'port': args.port, 'max_gib': args.max_gib}
    with (runtime/(args.name+'.stdout.log')).open('wb') as output, (runtime/(args.name+'.stderr.log')).open('wb') as errors:
        process = subprocess.Popen([shutil.which('powershell'), '-NoProfile', '-NonInteractive', '-File',
                                    str(root/'start.ps1'), '-Port', str(args.port), '-NoBrowser'],
                                   cwd=root, stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    parent = psutil.Process(process.pid)
    receipt['launcher_pid'] = process.pid
    try:
        last_report = 0
        while time.monotonic()-started < args.timeout:
            if process.poll() is not None:
                raise RuntimeError(f'start.ps1 exited: {process.returncode}')
            children = parent.children(recursive=True)
            usage = 0
            for child in [parent]+children:
                try:
                    usage += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, usage)
            if usage > args.max_gib*1024**3:
                raise RuntimeError('Process-tree RAM budget exceeded')
            if time.monotonic()-last_report > 15:
                print(f'{time.monotonic()-started:.0f}s, tree RSS {usage/1024**3:.2f} GiB, peak {peak/1024**3:.2f} GiB', flush=True)
                last_report = time.monotonic()
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/health', timeout=1) as response:
                    health = json.load(response)
                if health.get('ok'):
                    receipt.update(health=health, elapsed_s=time.monotonic()-started, peak_tree_rss_gib=peak/1024**3,
                                   child_pids=[c.pid for c in children], success=True)
                    print(json.dumps(receipt, ensure_ascii=False), flush=True)
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError('Startup timeout')
    except BaseException as error:
        receipt.update(success=False, error=str(error), peak_tree_rss_gib=peak/1024**3)
        for child in reversed(parent.children(recursive=True)):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        raise
    finally:
        (runtime/(args.name+'.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
