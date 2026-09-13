"""Disk-backed full corpus; bounded wave compilation and query working sets.

The full UTF-8 DCT is retained on disk. Conserved energies are compiled once;
only packets containing requested display coordinates need inverse transforms.
FTS selects passages for the existing information decoder: this is explicit
retrieval-augmented generation, not a globally compiled language model.
"""
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading

import numpy as np
from scipy.fft import idct

from .codec import encode_text, phase_angles
from .features import tokenize
from .memory import Document, WaveMemory
from .parallel import map as parallel_map
from .progress import phase, report

PASSAGE_CHARS = 4000
WORKING_CHARS = 120_000
BATCH_BYTES = 2 * 1024**2


def connect(path):
    connection = sqlite3.connect(path, timeout=60)
    connection.execute('PRAGMA cache_size=-8192')
    connection.execute('PRAGMA temp_store=FILE')
    return connection


def _compile_record(row):
    sequence, identifier, text, source = row
    packet = encode_text(text)
    metadata = WaveMemory._metadata_packet(Document(identifier, text, source))
    energy = physical = 0.0
    for item in (packet, metadata):
        power = item.coefficients**2
        energy += float(power.sum())
        physical += float(.5*np.dot((2*np.pi*30*np.arange(len(power))/len(power))**2, power))
    return (row, packet.coefficients.tobytes(), metadata.coefficients.tobytes(),
            packet.byte_length, metadata.byte_length, energy, physical)


class DiskDocuments(Sequence):
    def __init__(self, path, maximum, count):
        self.path, self.maximum, self.count = path, maximum, count

    def __len__(self):
        return self.count

    def __iter__(self):
        with closing(connect(self.path)) as connection:
            for row in connection.execute('SELECT id,text,source FROM documents WHERE sequence<=? ORDER BY sequence', (self.maximum,)):
                yield Document(*row)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.count)
            if step != 1:
                raise ValueError('Disk document slices require step=1')
        else:
            start = key if key >= 0 else self.count + key
            if not 0 <= start < self.count:
                raise IndexError(key)
            stop = start+1
        with closing(connect(self.path)) as connection:
            rows = connection.execute('SELECT id,text,source FROM documents WHERE sequence<=? ORDER BY sequence LIMIT ? OFFSET ?',
                                      (self.maximum, max(0, stop-start), start))
            result = [Document(*row) for row in rows]
        return result if isinstance(key, slice) else result[0]


class PagedWaveMemory(WaveMemory):
    paged = True

    def __init__(self, store):
        self.store = store
        for key, value in store.configuration().items():
            setattr(self, key, value)
        with store._transaction() as connection:
            self.store_revision = store._revision(connection)
            count, self.maximum = connection.execute('SELECT count(*),coalesce(max(sequence),0) FROM documents').fetchone()
            reset = connection.execute("SELECT value FROM metadata WHERE key='information_reset_revision'").fetchone()
        self.documents = DiskDocuments(store.path, self.maximum, count)
        from . import codec
        version = hashlib.sha256(Path(codec.__file__).read_bytes()+b'paged-v1').hexdigest()[:16]
        self.cache_dir = store.path.parent / '.compiler-cache' / store.path.name / f'paged-{version}-{reset[0] if reset else 0}'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / 'corpus.sqlite3'
        self._selection_lock = threading.RLock()
        self._display_lock = threading.RLock()
        self._display = None
        self._display_points = None
        self._working = OrderedDict()
        self._query_keys = OrderedDict()
        self._compiler_status = {'state': 'indexing', 'cache': 'pending', 'scope': 'paged_full_corpus'}
        with phase('Vollbestand: Schwingungs- und Textindex auf Festplatte'):
            self._build()
        self._compiler_status = {'state': 'ready', 'cache': 'hit' if self.compiled == 0 else 'incremental',
                                 'scope': 'paged_full_corpus', 'compiled_documents': self.compiled,
                                 'reused_documents': count-self.compiled}

    def _build(self):
        with closing(connect(self.cache_path)) as cache:
            cache.executescript('''
                CREATE TABLE IF NOT EXISTS packets(sequence INTEGER PRIMARY KEY, end_mode INTEGER NOT NULL,
                    text_coeff BLOB NOT NULL, meta_coeff BLOB NOT NULL, text_bytes INTEGER, meta_bytes INTEGER,
                    energy REAL, physical REAL);
                CREATE INDEX IF NOT EXISTS packet_end ON packets(end_mode);
                CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(document_id UNINDEXED, sequence UNINDEXED,
                    title, text, source UNINDEXED, tokenize='unicode61 remove_diacritics 2');
            ''')
            last, offset = cache.execute('SELECT coalesce(max(sequence),0),coalesce(max(end_mode),0) FROM packets').fetchone()
            reused = cache.execute('SELECT count(*) FROM packets WHERE sequence<=?', (self.maximum,)).fetchone()[0]
            self.compiled = 0
            report('Cache geprüft; vorhandene Dokumente werden wiederverwendet', reused, len(self.documents))
            with self.store._transaction() as source:
                if self.store._revision(source) != self.store_revision:
                    raise RuntimeError('Memory während des Starts geändert; Start wiederholen.')
                # A cheap append-only cursor; committed batches survive interruption.
                cursor = source.execute('SELECT sequence,id,text,source FROM documents WHERE sequence>? AND sequence<=? ORDER BY sequence',
                                        (last, self.maximum))
                batch, size = [], 0
                def commit_batch(batch):
                    nonlocal offset
                    estimate = sum(len(row[2].encode('utf-8')) for row in batch)*12 + 64*1024**2
                    if shutil.disk_usage(self.cache_dir).free < estimate:
                        raise OSError('Zu wenig freier Speicherplatz für den nächsten Cache-Block.')
                    results = parallel_map(_compile_record, batch, min_items=8)
                    with cache:
                        for row, text_blob, meta_blob, text_bytes, meta_bytes, energy, physical in results:
                            sequence, identifier, text, provenance = row
                            offset += (len(text_blob)+len(meta_blob))//8
                            cache.execute('INSERT INTO packets VALUES (?,?,?,?,?,?,?,?)',
                                          (sequence, offset, text_blob, meta_blob, text_bytes, meta_bytes, energy, physical))
                            title = text.split('\n', 1)[0][:200]
                            cache.executemany('INSERT INTO passages(document_id,sequence,title,text,source) VALUES (?,?,?,?,?)',
                                ((identifier, sequence, title, text[start:start+PASSAGE_CHARS], provenance)
                                 for start in range(0, max(1, len(text)), PASSAGE_CHARS)))
                    self.compiled += len(batch)
                    report('Vollbestand compiliert', reused+self.compiled, len(self.documents))
                for row in cursor:
                    if len(row[2].encode('utf-8')) > 16*1024**2:
                        raise ValueError(f'Dokument {row[1]} überschreitet 16 MiB; vor dem Import in Informationsabschnitte aufteilen.')
                    batch.append(row)
                    size += len(row[2].encode('utf-8'))
                    if size >= BATCH_BYTES or len(batch) >= 256:
                        commit_batch(batch)
                        batch, size = [], 0
                if batch:
                    commit_batch(batch)
            self.mode_count, self.text_bytes, self.metadata_bytes, self.energy, self.physical = cache.execute(
                'SELECT coalesce(max(end_mode),0),coalesce(sum(text_bytes),0),coalesce(sum(meta_bytes),0),'
                'coalesce(sum(energy),0),coalesce(sum(physical),0) FROM packets WHERE sequence<=?', (self.maximum,)).fetchone()
        if self.store.revision() != self.store_revision:
            raise RuntimeError('Memory während des Starts geändert; Start wiederholen (Cache bleibt erhalten).')

    def snapshot(self, time_s, points=256):
        if points < 2 or not np.isfinite(time_s):
            raise ValueError('At least two display points and a finite time required')
        indices = np.linspace(0, max(0, self.mode_count-1), min(points, self.mode_count)).astype(np.int64)
        with self._display_lock:
            if self._display_points != points:
                self._display = self._prepare_projection(indices)
                self._display_points = points
            projected = self._display.snapshot(time_s) if self._display is not None else None
        if projected is not None:
            q, p = projected
        else:
            q, p = self._stream_display(time_s, indices)
        return {'time_s': float(time_s), 'displacement': q.tolist(), 'quadrature': p.tolist(),
                'energy': self.energy, 'physical_energy': self.physical,
                'modal_count': self.mode_count, 'display_points': len(indices),
                'text_mode_count': self.mode_count-self.metadata_bytes, 'metadata_mode_count': self.metadata_bytes,
                'wave_layout': 'append_only_addressed_documents', 'document_count': len(self.documents),
                'dimensions': self.dimensions, 'key_mode_count': 0, 'key_energy': 0.0,
                'stored_bytes': self.text_bytes+self.metadata_bytes,
                'feature_mode': self.feature_mode, 'retrieval_policy': 'fts_passages',
                'storage': 'disk', 'energy_evaluation': 'conserved_full_corpus',
                'search_spectra': 'not_materialized'}

    def _prepare_projection(self, indices):
        from .projected_wave import ProjectedWave
        entries, total = [], 0
        previous_end = -1
        with closing(connect(self.cache_path)) as connection:
            for index in indices:
                if index < previous_end:
                    continue
                end, text, meta = connection.execute('SELECT end_mode,text_coeff,meta_coeff FROM packets '
                    'WHERE end_mode>? ORDER BY end_mode LIMIT 1', (int(index),)).fetchone()
                previous_end = end
                start = end-(len(text)+len(meta))//8
                for blob in (text, meta):
                    coefficients = np.frombuffer(blob, dtype=np.float64)
                    stop = start+len(coefficients)
                    selected = np.flatnonzero((indices >= start) & (indices < stop))
                    total += len(selected)*len(blob)
                    # Bound resident display work independently of future corpus size.
                    if total > 64*1024**2:
                        return None
                    if len(selected):
                        entries.append((coefficients, indices[selected]-start, selected))
                    start = stop
        return ProjectedWave(entries, len(indices))

    def _stream_display(self, time_s, indices):
        q, p = np.empty(len(indices)), np.empty(len(indices))
        with closing(connect(self.cache_path)) as connection:
            previous = None
            for position, index in enumerate(indices):
                if previous is None or index >= previous[0]:
                    previous = connection.execute('SELECT end_mode,text_coeff,meta_coeff FROM packets WHERE end_mode>? ORDER BY end_mode LIMIT 1', (int(index),)).fetchone()
                    end, text, meta = previous
                    start = end-(len(text)+len(meta))//8
                    for blob in (text, meta):
                        coefficients = np.frombuffer(blob, dtype=np.float64)
                        stop = start+len(coefficients)
                        selected = np.flatnonzero((indices >= start) & (indices < stop))
                        if len(selected):
                            angle = phase_angles(len(coefficients), time_s)
                            q[selected] = idct(coefficients*np.cos(angle), type=2, norm='ortho')[indices[selected]-start]
                            p[selected] = idct(coefficients*np.sin(angle), type=2, norm='ortho')[indices[selected]-start]
                        start = stop
        from .compute import record_cpu_snapshot
        record_cpu_snapshot()
        return q, p

    def working_memory(self, prompt, context=None):
        from .information import QUERY_FILLERS, QUESTION_PREDICATES
        from .features import canonical_term
        previous = (context or {}).get('unpaired', {})
        if not prompt and previous.get('working_set_key') in self._working:
            return self._working[previous['working_set_key']]
        query_key = json.dumps([prompt, previous.get('subject'), previous.get('subjects'), previous.get('focus')], ensure_ascii=False)
        cached_key = self._query_keys.get(query_key)
        if cached_key in self._working:
            self._working.move_to_end(cached_key)
            return self._working[cached_key]
        prompt_terms = list(dict.fromkeys(tokenize(prompt)))[:16]
        domain = [term for term in prompt_terms if canonical_term(term) not in QUERY_FILLERS | QUESTION_PREDICATES]
        prior_terms = []
        # Persisted information subjects resolve follow-ups without arbitrary conversation text.
        for key in ('subject', 'subjects', 'focus'):
            value = previous.get(key, '')
            if isinstance(value, str):
                prior_terms.extend(tokenize(value))
            elif isinstance(value, list):
                prior_terms.extend(tokenize(' '.join(str(item) for item in value)))
        terms = list(dict.fromkeys(domain or prior_terms or prompt_terms))[:24]
        with closing(connect(self.cache_path)) as connection:
            rows = []
            if terms:
                match = ' OR '.join('"'+term.replace('"', '""')+'"' for term in terms)
                sql = ('SELECT document_id,title,text,source,rowid FROM passages WHERE passages MATCH ? '
                       'AND sequence<=? ORDER BY bm25(passages,0,0,8,1,0),rowid LIMIT 24')
                # A follow-up facet must not replace its subject with a generic
                # article about "unit", "formula", etc. Prefer their intersection.
                facets = [term for term in prompt_terms if term not in terms]
                if facets:
                    facet_match = ' OR '.join('"'+term.replace('"', '""')+'"' for term in facets)
                    rows = connection.execute(sql, ('('+match+') AND ('+facet_match+')', self.maximum)).fetchall()
                if not rows:
                    rows = connection.execute(sql, (match, self.maximum)).fetchall()
        # Information-only authored grammar facts are small; never bring Wikipedia into this base.
        with closing(connect(self.store.path)) as connection:
            base = []
            base_chars = 0
            for prefix in ('conversation-fact-', 'conversation-lexicon-'):
                for row in connection.execute('SELECT id,text,source FROM documents WHERE id>=? AND id<? '
                        'AND sequence<=? AND length(text)<=? ORDER BY id LIMIT 128',
                        (prefix, prefix+'\uffff', self.maximum, PASSAGE_CHARS)):
                    if base_chars + len(row[1]) > WORKING_CHARS//3:
                        break
                    base.append(Document(*row))
                    base_chars += len(row[1])
        selected = list(base)
        remaining = WORKING_CHARS-sum(len(doc.text) for doc in base)
        for identifier, title, text, source, rowid in rows:
            if len(text)+len(title)+2 > remaining:
                break
            selected.append(Document(f'{identifier}:passage:{rowid}', title+'\n\n'+text, source))
            remaining -= len(text)+len(title)+2
        key = hashlib.sha256(json.dumps([(d.id,d.text,d.source) for d in selected],ensure_ascii=False).encode()).hexdigest()
        with self._selection_lock:
            if key not in self._working:
                # The generator needs only records, not a second wave/diagnostic index.
                working = object.__new__(WaveMemory)
                working.documents = selected
                working._paged_key = key
                working._compiler_cache_dir = self.cache_dir / f'query-{int(key[:8],16)%8}'
                self._working[key] = working
                while len(self._working) > 2:
                    self._working.popitem(last=False)
            self._working.move_to_end(key)
            self._query_keys[query_key] = key
            while len(self._query_keys) > 8:
                self._query_keys.popitem(last=False)
            return self._working[key]

    def save(self, path):
        raise ValueError('Große Memory bleibt in SQLite; monolithischer NPZ-Export würde das RAM-Limit überschreiten.')

    def ask(self, prompt, top_k=1, time_s=0.0):
        from .unpaired_runtime import respond_wave
        return respond_wave(self, prompt, top_k=top_k, time_s=time_s)[0]

    def with_documents_added(self, documents):
        raise ValueError('Im Festplattenmodus Dokumente über MemoryStore oder WaveEngine hinzufügen.')

    def scores(self, prompt, time_s=0.0):
        raise ValueError('Der große Bestand verwendet einen Textindex statt einer dichten Suchspektrenmatrix.')
