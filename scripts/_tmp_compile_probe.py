"""Temporary probe: payload sizes and cache feasibility of the compiled model."""
import sys, time, io, pickle, gzip, json

sys.path.insert(0, '.')
import numpy as np

from freqai.store import MemoryStore

store = MemoryStore('memory/memory.sqlite3')
rev, docs = store.snapshot()
records = [{'id': d.id, 'text': d.text, 'source': d.source} for d in docs]

from freqai.unpaired import UnpairedWaveModel

t = time.perf_counter()
model = UnpairedWaveModel(records, order=2)
print('compile', round(time.perf_counter() - t, 2), flush=True)
stats = model.storage_stats()
print('stats', json.dumps(stats), flush=True)
print('corpus_digest', model.corpus_digest, flush=True)

lengths = np.array([len(s) for s in model.supports.values()], dtype=np.int64)
print('fields', len(lengths), 'edge_total', int(lengths.sum()),
      'share_len<=2', float((lengths <= 2).mean()),
      'max', int(lengths.max()), 'p99', int(np.percentile(lengths, 99)), flush=True)

t = time.perf_counter()
buf = io.BytesIO()
pickle.dump(model, buf, protocol=pickle.HIGHEST_PROTOCOL)
raw = buf.getvalue()
print('pickle_seconds', round(time.perf_counter() - t, 2), 'pickle_bytes', len(raw), flush=True)
t = time.perf_counter()
buf2 = io.BytesIO(raw)
buf2.seek(0)
reloaded = pickle.load(buf2)
print('reload_seconds', round(time.perf_counter() - t, 2), flush=True)
t = time.perf_counter()
answer = reloaded.generate('Wer ist Albert Einstein gewesen?')
print('reload_answer', repr(answer['text']), round(time.perf_counter() - t, 2), flush=True)
print('signature_equal',
      model.model_signature()['corpus_digest'] == reloaded.model_signature()['corpus_digest'], flush=True)
