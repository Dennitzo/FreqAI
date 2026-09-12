"""Optional float64 CUDA batches with a NumPy/SciPy multicore fallback.

Python corpus parsing stays in the process pool. CUDA is used only in the main
process, for sufficiently large numerical batches. Status reports completed
operations, not a claim of continuous utilization of every hardware core.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import multiprocessing
import os
import threading

import numpy as np
from scipy import fft as scipy_fft

from .parallel import chunk_ranges, worker_count

_lock = threading.RLock()
_cupy = None
_devices = []
_probed = False
_failure = None
_executor = None
_device_streams = {}
_device_locks = {}
_operations = Counter()
_device_operations = Counter()
_last_backend = None
_last_backend_by_operation = {}
_MIN_CUDA_ELEMENTS = 262144


def _requested_backend():
    backend = os.environ.get("FREQAI_COMPUTE", "auto").strip().lower()
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("FREQAI_COMPUTE must be auto, cpu, or cuda")
    return backend


def compute_status() -> dict:
    """Read the existing backend state without imports, probes, or GPU work."""
    with _lock:
        return {"requested_backend": _requested_backend(), "cuda_probed": _probed,
                "cuda_available": bool(_devices) and _failure is None if _probed else None,
                "cuda_devices": copy.deepcopy(_devices), "fallback_reason": _failure,
                "last_backend": _last_backend, "precision": "float64/complex128",
                "last_backend_by_operation": dict(_last_backend_by_operation),
                "cpu_workers": worker_count(), "cuda_min_elements": _MIN_CUDA_ELEMENTS,
                "completed_operations": dict(_operations),
                "device_completed_operations": dict(_device_operations)}


def _disable_cuda(error):
    global _failure
    with _lock:
        _failure = f"{type(error).__name__}: {error}"


def _probe_cuda():
    global _probed, _cupy, _devices, _executor
    with _lock:
        if _probed:
            return bool(_devices) and _failure is None
        _probed = True
        try:
            import cupy as cp
            selected = os.environ.get("FREQAI_CUDA_DEVICES", "all").strip().lower()
            count = cp.cuda.runtime.getDeviceCount()
            ids = list(range(count)) if selected == "all" else list(dict.fromkeys(
                int(value.strip()) for value in selected.split(",")))
            if not ids or any(index < 0 or index >= count for index in ids):
                raise ValueError("No valid CUDA devices selected")
            devices = []
            for index in ids:
                with cp.cuda.Device(index):
                    # Probe both an actual kernel and cuFFT, not driver presence.
                    probe = cp.fft.fft(cp.arange(16, dtype=cp.float64))
                    if not np.isfinite(probe.get()).all():
                        raise RuntimeError("CUDA numerical probe returned nonfinite data")
                    props = cp.cuda.runtime.getDeviceProperties(index)
                    name = props["name"]
                    devices.append({"id": index,
                                    "name": name.decode() if isinstance(name, bytes) else str(name),
                                    "total_memory_bytes": int(props["totalGlobalMem"])})
                    # Reuse one stream arena per device. Creating a new stream
                    # each frame can retain many CuPy memory-pool arenas.
                    _device_streams[index] = cp.cuda.Stream(non_blocking=True)
                    _device_locks[index] = threading.RLock()
            _cupy, _devices = cp, devices
            _executor = ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="freqai-cuda")
        except Exception as error:
            _disable_cuda(error)
            return False
        return True


def cuda_for(elements: int, *, backend: str | None = None) -> bool:
    """Select CUDA lazily; small arrays and spawned CPU workers stay on CPU."""
    requested = _requested_backend() if backend is None else backend
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be auto, cpu, or cuda")
    if requested == "cpu" or multiprocessing.current_process().name != "MainProcess":
        return False
    if elements < _MIN_CUDA_ELEMENTS and requested != "cuda":
        return False
    return _probe_cuda()


def _completed(backend, kind, device_ids=()):
    global _last_backend
    with _lock:
        _last_backend = backend
        _last_backend_by_operation[kind] = backend
        _operations[f"{backend}_{kind}"] += 1
        for index in device_ids:
            _device_operations[str(index)] += 1


def record_cpu_snapshot():
    """Record a completed reference snapshot after the caller evaluated it."""
    _completed("cpu", "wave_snapshot")


def fft_rows(values: np.ndarray, *, backend: str | None = None) -> np.ndarray:
    """Orthonormal FFT over independent rows, preserving complex128 precision."""
    values = np.asarray(values)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("FFT batch must have shape (rows, nonzero modes)")
    values = np.asarray(values, dtype=np.complex128 if values.dtype.kind == "c" else np.float64)
    requested = _requested_backend() if backend is None else backend
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be auto, cpu, or cuda")
    # These inputs and outputs reside on the host. Measured PCIe transfers cost
    # more than the multicore CPU FFT here; auto reserves CUDA for resident wave
    # dynamics. Explicit cuda remains available for tests and other hardware.
    if values.shape[0] and requested == "cuda" and cuda_for(values.size, backend=requested):
        cp = _cupy

        def transform(index, start, stop):
            with _device_locks[index], cp.cuda.Device(index), _device_streams[index]:
                result = np.empty((stop-start, values.shape[1]), dtype=np.complex128)
                # Bound temporary buffers independently of corpus size/VRAM.
                batch_rows = max(1, (32 * 1024**2) // (values.shape[1] * 16))
                for begin in range(start, stop, batch_rows):
                    end = min(stop, begin+batch_rows)
                    data = cp.asarray(values[begin:end], dtype=cp.complex128)
                    result[begin-start:end-start] = cp.asnumpy(cp.fft.fft(data, axis=-1, norm="ortho"))
                return result

        ranges = chunk_ranges(len(values), len(_devices))
        ids = [device["id"] for device in _devices[:len(ranges)]]
        futures = [_executor.submit(transform, index, start, stop)
                   for index, (start, stop) in zip(ids, ranges)]
        # Drain every submitted task before falling back or freeing its inputs.
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(error)
        if not errors:
            _completed("cuda", "fft", ids)
            return np.concatenate(results, axis=0)
        _disable_cuda(errors[0])
    result = scipy_fft.fft(values, axis=-1, norm="ortho", workers=worker_count())
    _completed("cpu", "fft")
    return result


class AddressedWaveAccelerator:
    """Resident CUDA state for the immutable, addressed coefficient bank.

    Every mode contributes to the evolving quadratures and both energy sums.
    Requested spatial display coordinates evaluate the full orthonormal inverse
    DCT sum of their packet. Unrequested pixels need not be materialized; no
    modes, packets, or addressed channels are dropped or mixed.
    """

    def __init__(self, coefficients, relative_frequencies, sizes, groups):
        self.coefficients = coefficients
        self.relative_frequencies = relative_frequencies
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.sizes)))
        self.groups = groups
        self._lock = threading.RLock()
        self._shards = None
        self._display_key = None
        self._display = None

    def _prepare(self):
        cp = _cupy
        phase_addresses = np.empty(len(self.coefficients), dtype=np.int32)
        offset = 0
        for size, addresses in self.groups:
            phase_addresses[addresses] = offset + np.arange(size, dtype=np.int32)
            offset += size
        self._phase_addresses = phase_addresses
        shards = []
        for device, (start, stop) in zip(_devices, chunk_ranges(len(self.coefficients), len(_devices))):
            with cp.cuda.Device(device["id"]):
                shards.append({"id": device["id"],
                               "coefficients": cp.asarray(self.coefficients[start:stop]),
                               "phase_addresses": cp.asarray(phase_addresses[start:stop]),
                               "omega_squared": cp.asarray((2*np.pi*30*self.relative_frequencies[start:stop])**2)})
                cp.cuda.get_current_stream().synchronize()
        self._shards = shards

    def _prepare_display(self, indices):
        cp = _cupy
        packet_ids = np.searchsorted(self.starts[1:], indices, side="right")
        local_positions = indices - self.starts[packet_ids]
        width = int(self.sizes[packet_ids].max())
        coefficients = np.zeros((len(indices), width))
        phases = np.zeros((len(indices), width), dtype=np.int32)
        basis = np.zeros((len(indices), width))
        for row, (packet_id, position) in enumerate(zip(packet_ids, local_positions)):
            start, stop = self.starts[packet_id:packet_id+2]
            size = int(stop-start)
            coefficients[row, :size] = self.coefficients[start:stop]
            phases[row, :size] = self._phase_addresses[start:stop]
            basis[row, :size] = np.sqrt(2.0/size)*np.cos(np.pi*(position+.5)*np.arange(size)/size)
            basis[row, 0] = 1.0/np.sqrt(size)
        display = []
        for shard, (start, stop) in zip(self._shards, chunk_ranges(len(indices), len(self._shards))):
            with cp.cuda.Device(shard["id"]):
                display.append({"id": shard["id"], "coefficients": cp.asarray(coefficients[start:stop]),
                                "phase_addresses": cp.asarray(phases[start:stop]),
                                "basis": cp.asarray(basis[start:stop])})
                cp.cuda.get_current_stream().synchronize()
        self._display = display
        self._display_key = tuple(int(index) for index in indices)

    def snapshot(self, time_s, indices):
        """Return a CUDA result, or None when the regular CPU path should run."""
        if not len(indices) or not cuda_for(len(self.coefficients)):
            return None
        from .codec import phase_angles
        cp = _cupy
        with self._lock:
            try:
                packet_ids = np.searchsorted(self.starts[1:], indices, side="right")
                # Limit both host and device display matrices. A huge packet
                # or an export of every pixel retains the full CPU FFT path.
                if len(indices)*int(self.sizes[packet_ids].max()) > 2_000_000:
                    return None
                if self._shards is None:
                    self._prepare()
                if self._display_key != tuple(int(index) for index in indices):
                    self._prepare_display(indices)
                # CPU phase reduction preserves the existing long-time clock
                # exactly. Each length's phase grid is transferred only once.
                angles = np.concatenate([phase_angles(size, time_s) for size, _ in self.groups])

                def evaluate(shard, display):
                    with _device_locks[shard["id"]], cp.cuda.Device(shard["id"]), _device_streams[shard["id"]]:
                        theta = cp.asarray(angles)
                        cosine, sine = cp.cos(theta), cp.sin(theta)
                        q = shard["coefficients"]*cosine[shard["phase_addresses"]]
                        p = shard["coefficients"]*sine[shard["phase_addresses"]]
                        norm = q*q+p*p
                        energy = cp.stack((cp.sum(norm), .5*cp.sum(shard["omega_squared"]*norm)))
                        if display is None:
                            return np.zeros(0), np.zeros(0), cp.asnumpy(energy)
                        weighted = display["coefficients"]*display["basis"]
                        spatial_q = cp.sum(weighted*cosine[display["phase_addresses"]], axis=-1)
                        spatial_p = cp.sum(weighted*sine[display["phase_addresses"]], axis=-1)
                        return cp.asnumpy(spatial_q), cp.asnumpy(spatial_p), cp.asnumpy(energy)

                display_by_id = {item["id"]: item for item in self._display}
                futures = [_executor.submit(evaluate, shard, display_by_id.get(shard["id"]))
                           for shard in self._shards]
                results, errors = [], []
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as error:
                        errors.append(error)
                if errors:
                    raise errors[0]
                q = np.concatenate([result[0] for result in results])
                p = np.concatenate([result[1] for result in results])
                energy = np.sum([result[2] for result in results], axis=0)
                if not (np.isfinite(q).all() and np.isfinite(p).all() and np.isfinite(energy).all()):
                    raise ValueError("CUDA wave snapshot returned nonfinite data")
                _completed("cuda", "wave_snapshot", [shard["id"] for shard in self._shards])
                return {"displacement": q.tolist(), "quadrature": p.tolist(),
                        "energy": float(energy[0]), "physical_energy": float(energy[1])}
            except Exception as error:
                self._shards = self._display = None
                _disable_cuda(error)
                return None


__all__ = ["AddressedWaveAccelerator", "compute_status", "cuda_for", "fft_rows", "record_cpu_snapshot"]
