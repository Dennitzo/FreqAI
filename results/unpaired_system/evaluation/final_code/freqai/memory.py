"""Addressed standing-wave payloads with coherent Fourier interference lookup.

Address channels are deliberately retained: summing every text into the same
unlabelled modes destroys separability. Lexical document lookup is a numerical
diagnostic only; the application generates through unpaired_runtime.
"""

from dataclasses import asdict, dataclass
from collections import Counter
import copy
import json
from pathlib import Path
import os
import tempfile

import numpy as np
from scipy.fft import idct

from .codec import WavePacket, decode_text, encode_text, phase_angles, sample_wave
from .features import content_terms, feature_vector, text_spectrum, tokenize


@dataclass(frozen=True)
class Document:
    """Information record; prompt exists only to preserve historical archives.

    Active storage and language generation reject a nonempty legacy prompt.
    Diagnostic indexing uses text, never that historical field.
    """
    id: str
    text: str
    source: str = ""
    prompt: str = ""


class WaveMemory:
    def __init__(self, documents: list[Document], dimensions: int = 4096,
                 feature_mode: str = "hybrid", min_score: float = 0.18,
                 retrieval_policy: str = "legacy", min_coverage: float = 0.6):
        if not 0 <= min_score <= 1 or not np.isfinite(min_score):
            raise ValueError("min_score must be between zero and one")
        if not isinstance(dimensions, int) or not 16 <= dimensions <= 262144:
            raise ValueError("dimensions must be an integer between 16 and 262144")
        if feature_mode not in {"raw_bytes", "words", "hybrid", "morphology"}:
            raise ValueError("Invalid feature mode")
        if retrieval_policy not in {"legacy", "coverage"}:
            raise ValueError("Invalid retrieval policy")
        if not np.isfinite(min_coverage) or not 0 <= min_coverage <= 1:
            raise ValueError("min_coverage must be between zero and one")
        self.dimensions = dimensions
        self.feature_mode = feature_mode
        self.min_score = float(min_score)
        self.retrieval_policy = retrieval_policy
        self.min_coverage = float(min_coverage)
        self.documents = list(documents)
        if any(not isinstance(d, Document) or not isinstance(d.id, str) or not d.id or
               not isinstance(d.text, str) or not isinstance(d.source, str) or
               not isinstance(d.prompt, str) for d in self.documents):
            raise ValueError("Each document needs a nonempty ID and text")
        if len({d.id for d in self.documents}) != len(self.documents):
            raise ValueError("Document IDs must be unique")
        self._rebuild()

    def _rebuild(self) -> None:
        self.payloads = [encode_text(d.text) for d in self.documents]
        self.metadata_payloads = [self._metadata_packet(d) for d in self.documents]
        self.spectra = np.array([text_spectrum(d.text, self.dimensions, self.feature_mode)
                                 for d in self.documents], dtype=np.complex128).reshape(-1, self.dimensions)
        self._words = [set(tokenize(d.text)) for d in self.documents]
        self._terms = [content_terms(d.text) for d in self.documents]
        self._term_counts = Counter(term for terms in self._terms for term in terms)
        self._prepare_arrays()

    @staticmethod
    def _metadata_packet(document: Document) -> WavePacket:
        metadata = {"id": document.id, "source": document.source}
        if document.prompt:
            metadata["prompt"] = document.prompt  # Historical bytes only.
        return encode_text(json.dumps(metadata,
                                      ensure_ascii=False, separators=(",", ":")))

    @property
    def archive(self) -> WavePacket:
        """Legacy export codec only; not the state of the continuously running wave."""
        if self._archive is None:
            serialized = json.dumps([asdict(d) for d in self.documents], ensure_ascii=False, separators=(",", ":"))
            self._archive = encode_text(serialized)
        return self._archive

    def _prepare_arrays(self) -> None:
        self._archive = None
        self._spectra_conjugate = np.conj(self.spectra)
        self._key_power = np.sum(np.abs(self.spectra)**2, axis=0)
        self._packets = [packet for pair in zip(self.payloads, self.metadata_payloads) for packet in pair]
        self._sizes = [len(packet.coefficients) for packet in self._packets]
        self._coefficients = np.concatenate([p.coefficients for p in self._packets]) if self._packets else np.zeros(0)
        self._relative_frequencies = np.concatenate([np.arange(size) / size for size in self._sizes]) if self._sizes else np.zeros(0)
        # Equal-length packets share a phase grid and can be transformed as a
        # batch. Addresses still refer to their original append-only positions.
        starts_by_size = {}
        offset = 0
        for size in self._sizes:
            starts_by_size.setdefault(size, []).append(offset)
            offset += size
        self._wave_groups = [(size, np.asarray(starts, dtype=np.intp)[:, None] + np.arange(size))
                             for size, starts in starts_by_size.items()]
        for packet in self._packets:
            packet.coefficients.flags.writeable = False

    def with_documents_added(self, documents: list[Document]) -> "WaveMemory":
        """Prepare an immutable append without re-encoding any existing document.

        The caller can persist the candidate first, then atomically publish it.
        Existing packet objects, frequencies and time origins stay unchanged.
        """
        documents = list(documents)
        existing = {d.id for d in self.documents}
        for document in documents:
            if (not isinstance(document, Document) or not isinstance(document.id, str) or not document.id
                    or not isinstance(document.text, str) or not isinstance(document.source, str)
                    or not isinstance(document.prompt, str)):
                raise ValueError("Each document needs a nonempty string ID, text and source")
            if document.id in existing:
                raise ValueError("Document ID already exists")
            existing.add(document.id)
        if not documents:
            return self
        added_payloads = [encode_text(d.text) for d in documents]
        added_metadata = [self._metadata_packet(d) for d in documents]
        added_spectra = np.stack([text_spectrum(d.text, self.dimensions, self.feature_mode) for d in documents])
        candidate = copy.copy(self)
        candidate.documents = self.documents + documents
        candidate.payloads = self.payloads + added_payloads
        candidate.metadata_payloads = self.metadata_payloads + added_metadata
        candidate.spectra = np.concatenate((self.spectra, added_spectra), axis=0)
        candidate._words = self._words + [set(tokenize(d.text)) for d in documents]
        new_terms = [content_terms(d.text) for d in documents]
        candidate._terms = self._terms + new_terms
        candidate._term_counts = self._term_counts.copy()
        candidate._term_counts.update(term for terms in new_terms for term in terms)
        candidate._prepare_arrays()
        return candidate

    def scores(self, prompt: str, time_s: float = 0.0) -> np.ndarray:
        if not np.isfinite(time_s):
            raise ValueError("Time must be finite")
        query = text_spectrum(prompt, self.dimensions, self.feature_mode)
        # Readout of the coherent cross term. Probe and memory share a clock.
        # This is algebraically half the energy change of K_i + Q; evaluate the
        # cross term directly to avoid catastrophic cancellation near zero.
        # U(t)* U(t) = I: cancel the shared phase analytically, avoiding two
        # M-by-D temporal arrays on each query. The inner product is still
        # evaluated on actual complex Fourier coefficients, not original text.
        return np.real(np.einsum("ij,j->i", self._spectra_conjugate, query, optimize=False))

    def coverage(self, prompt: str) -> np.ndarray:
        terms = content_terms(prompt, query=True)
        if not terms:
            return np.zeros(len(self.documents))
        weights = {term: 1.0 + np.log((1 + len(self.documents)) / (1 + self._term_counts[term]))
                   for term in terms}
        total = sum(weights.values())
        return np.array([sum(weights[t] for t in terms & doc_terms) / total for doc_terms in self._terms])

    def ask(self, prompt: str, top_k: int = 1, time_s: float = 0.0) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must contain text")
        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        scores = self.scores(prompt, time_s)
        query_words = set(tokenize(prompt))
        coverages = self.coverage(prompt) if self.retrieval_policy == "coverage" else None
        # Equal theoretical scores can differ by a few floating-point ulps as
        # the common phase evolves. Readout resolution is fixed at 1e-12;
        # corpus order breaks ties, independent of time. Report raw scores.
        readout_scores = np.round(scores, decimals=12)
        order = np.argsort(-readout_scores, kind="stable")
        matches = []
        for index in order:
            score = float(scores[index])
            if readout_scores[index] < self.min_score:
                continue
            # Fixed conservative lexical guard: feature-hash collisions alone
            # must not generate answers. This is visible in method documentation.
            if self.retrieval_policy == "coverage":
                if round(float(coverages[index]), 12) < self.min_coverage or not content_terms(prompt, query=True):
                    continue
            elif self.feature_mode == "morphology":
                if not (content_terms(prompt) & self._terms[index]):
                    continue
            elif self.feature_mode != "raw_bytes" and not (query_words & self._words[index]):
                continue
            doc = self.documents[index]
            text = decode_text(self.payloads[index], time_s)
            matches.append({"id": doc.id, "text": text, "source": doc.source,
                            "score": score, "interference": 2.0 * score})
            if coverages is not None:
                matches[-1]["query_coverage"] = float(coverages[index])
            if len(matches) >= top_k:
                break
        return {
            "answer": "\n\n".join(m["text"] for m in matches) if matches else
                      "Keine ausreichend passende gespeicherte Textstelle gefunden.",
            "abstained": not matches,
            "matches": matches,
            "method": "coherent_fourier_interference_then_inverse_standing_wave",
            "feature_mode": self.feature_mode,
            "retrieval_policy": self.retrieval_policy,
            "time_s": float(time_s),
            "note": "Extraktive Antwort aus dem Wellenspeicher; keine frei gelernte Sprachgenerierung.",
        }

    def snapshot(self, time_s: float, points: int = 256) -> dict:
        if points < 2 or not np.isfinite(time_s):
            raise ValueError("At least two display points and a finite time required")
        # Direct sum of addressed standing waves: adding a packet does not
        # retune or reset a single pre-existing mode. Evaluate ALL coefficients
        # before reducing the spatial display, including source/ID text modes.
        count = len(self._coefficients)
        indices = np.linspace(0, max(0, count-1), min(points, count)).astype(int)
        display_q = np.empty(len(indices))
        display_p = np.empty(len(indices))
        norm = np.empty_like(self._coefficients)
        # This is the same per-packet phase reduction and inverse DCT, including
        # all stored coefficients; batching never mixes different documents.
        for size, addresses in self._wave_groups:
            theta = phase_angles(size, time_s)
            coefficients = self._coefficients[addresses]
            grouped_q = coefficients * np.cos(theta)
            grouped_p = coefficients * np.sin(theta)
            norm[addresses] = grouped_q*grouped_q + grouped_p*grouped_p
            spatial_q = idct(grouped_q, type=2, norm="ortho", axis=-1)
            spatial_p = idct(grouped_p, type=2, norm="ortho", axis=-1)
            # All spatial samples were computed above. Only their displayed
            # coordinates need a long-lived output buffer, avoiding four full
            # corpus-sized q/p arrays on every frame of a large live memory.
            flat_addresses = addresses.ravel()
            positions = np.searchsorted(flat_addresses, indices)
            selected = np.flatnonzero(positions < len(flat_addresses))
            selected = selected[flat_addresses[positions[selected]] == indices[selected]]
            display_q[selected] = spatial_q.ravel()[positions[selected]]
            display_p[selected] = spatial_p.ravel()[positions[selected]]
        result = {"time_s": float(time_s), "displacement": display_q.tolist(),
                  "quadrature": display_p.tolist(), "energy": float(np.sum(norm)),
                  "physical_energy": float(.5*np.sum((2*np.pi*30*self._relative_frequencies)**2 * norm)),
                  "modal_count": count, "display_points": len(indices),
                  "text_mode_count": sum(len(p.coefficients) for p in self.payloads),
                  "metadata_mode_count": sum(len(p.coefficients) for p in self.metadata_payloads),
                  "wave_layout": "append_only_addressed_documents"}
        # Every key mode receives the same per-frequency clock rotation. The
        # precomputed power sum is exactly the sum over addressed channels and
        # avoids allocating the full M-by-D key bank on every display tick.
        key_phase = np.exp(1j * phase_angles(self.dimensions, time_s))
        result.update({"document_count": len(self.documents), "dimensions": self.dimensions,
                       "key_mode_count": int(self.spectra.size),
                       "key_energy": float(np.sum(self._key_power * np.abs(key_phase)**2)),
                       "stored_bytes": sum(p.byte_length for p in self._packets),
                       "feature_mode": self.feature_mode,
                       "retrieval_policy": self.retrieval_policy})
        return result

    def add_document(self, text: str, source: str = "user", document_id: str | None = None) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Document must contain text")
        if document_id is None:
            number = len(self.documents) + 1
            existing = {doc.id for doc in self.documents}
            while f"doc-{number}" in existing:
                number += 1
            document_id = f"doc-{number}"
        if any(doc.id == document_id for doc in self.documents):
            raise ValueError("Document ID already exists")
        replacement = self.with_documents_added([Document(document_id, text, str(source))])
        self.__dict__.update(replacement.__dict__)
        return document_id

    def save(self, path: str | Path) -> None:
        """Atomic, pickle-free file; corpus texts are stored as wave coefficients."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"format": "freqai-standing-v1", "byte_length": self.archive.byte_length,
                  "sha256": self.archive.sha256, "dimensions": self.dimensions,
                  "feature_mode": self.feature_mode, "min_score": self.min_score,
                  "retrieval_policy": self.retrieval_policy, "min_coverage": self.min_coverage}
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, coefficients=self.archive.coefficients,
                                    header=np.array(json.dumps(header)))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> "WaveMemory":
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"].item()))
            if header.get("format") != "freqai-standing-v1":
                raise ValueError("Unknown memory format")
            packet = WavePacket(np.array(data["coefficients"], dtype=np.float64),
                                int(header["byte_length"]), str(header["sha256"]))
        documents = [Document(**doc) for doc in json.loads(decode_text(packet))]
        return cls(documents, int(header["dimensions"]), str(header["feature_mode"]),
                   float(header["min_score"]), str(header.get("retrieval_policy", "legacy")),
                   float(header.get("min_coverage", 0.6)))

    def direct_cosine_scores(self, prompt: str) -> np.ndarray:
        """Reference baseline: Parseval requires agreement with wave scores."""
        query = feature_vector(prompt, self.dimensions, self.feature_mode)
        return np.array([np.dot(feature_vector(d.text, self.dimensions, self.feature_mode), query)
                         for d in self.documents])
