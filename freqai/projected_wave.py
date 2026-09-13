"""Bounded, exact inverse-DCT coordinates on CPU or all selected CUDA devices."""
import threading
import numpy as np
from . import compute
from .codec import phase_angles


class ProjectedWave:
    def __init__(self, entries, points):
        self.entries, self.points = [], points
        self._shards = None
        self._lock = threading.RLock()
        for coefficients, positions, outputs in entries:
            n = len(coefficients)
            k = np.arange(n, dtype=np.float64)
            basis = np.sqrt(2.0/n)*np.cos(np.pi*(np.asarray(positions)[:, None]+.5)*k/n)
            basis[:, 0] = 1/np.sqrt(n)
            self.entries.append((coefficients[None, :]*basis, np.asarray(outputs)))
        self.elements = sum(weights.size for weights, _ in self.entries)

    def snapshot(self, time_s):
        with self._lock:
            if compute.cuda_for(self.elements):
                try:
                    return self._cuda(time_s)
                except Exception as error:
                    self._shards = None
                    compute._disable_cuda(error)
            q, p = np.empty(self.points), np.empty(self.points)
            for weights, outputs in self.entries:
                angles = phase_angles(weights.shape[1], time_s)
                # Avoid starting a BLAS team for each tiny display reduction.
                q[outputs] = np.einsum('ij,j->i', weights, np.cos(angles))
                p[outputs] = np.einsum('ij,j->i', weights, np.sin(angles))
            compute.record_cpu_snapshot()
            return q, p

    def _cuda(self, time_s):
        cp = compute._cupy
        if self._shards is None:
            shards = []
            for position, device in enumerate(compute._devices):
                index = device['id']
                with cp.cuda.Device(index):
                    entries = [(cp.asarray(weights), cp.arange(weights.shape[1], dtype=cp.float64), outputs)
                               for weights, outputs in self.entries[position::len(compute._devices)]]
                    cp.cuda.get_current_stream().synchronize()
                    shards.append((index, entries))
            self._shards = shards

        def evaluate(shard):
            index, entries = shard
            with compute._device_locks[index], cp.cuda.Device(index), compute._device_streams[index]:
                qs, ps, outputs = [], [], []
                for weights, k, positions in entries:
                    cycles = np.remainder((30.0/len(k))*time_s, 1.0)
                    angle = 2*np.pi*cp.remainder(k*cycles, 1.0)
                    qs.append(cp.sum(weights*cp.cos(angle)[None, :], axis=1))
                    ps.append(cp.sum(weights*cp.sin(angle)[None, :], axis=1))
                    outputs.extend(positions)
                if not outputs:
                    return [], [], []
                return outputs, cp.asnumpy(cp.concatenate(qs)), cp.asnumpy(cp.concatenate(ps))

        futures = [compute._executor.submit(evaluate, shard) for shard in self._shards]
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
        q, p = np.empty(self.points), np.empty(self.points)
        for outputs, values_q, values_p in results:
            if len(outputs):
                q[outputs], p[outputs] = values_q, values_p
        if not (np.isfinite(q).all() and np.isfinite(p).all()):
            raise ValueError('Nonfinite projected CUDA wave')
        compute._completed('cuda', 'wave_snapshot', [index for index, entries in self._shards if entries])
        return q, p
