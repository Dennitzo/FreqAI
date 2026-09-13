"""Unbuffered terminal progress, including heartbeats for compiler phases."""
from contextlib import contextmanager
import sys
import threading
import time


def report(message, done=None, total=None):
    suffix = f" | {done:,}/{total:,} ({100*done/max(1,total):.1f}%)" if total is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}] {message}{suffix}", file=sys.stderr, flush=True)


@contextmanager
def phase(message):
    started = time.monotonic()
    stop = threading.Event()
    report(message)
    def heartbeat():
        while not stop.wait(10):
            report(f"{message} – läuft seit {time.monotonic()-started:.0f} s")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException as error:
        report(f"{message}: FEHLER {type(error).__name__}: {error}")
        raise
    else:
        report(f"{message}: fertig ({time.monotonic()-started:.1f} s)")
    finally:
        stop.set()
        thread.join()
