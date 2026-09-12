"""Addressed standing-wave payloads with coherent Fourier interference lookup.

Address channels are deliberately retained: summing every text into the same
unlabelled modes destroys separability. Retrieval is lexical and extractive.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import os
import tempfile

import numpy as np

from .codec import WavePacket, decode_text, encode_text, phase_angles, sample_wave
from .features import feature_vector, text_spectrum, tokenize


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = ""


class WaveMemory:
    def __init__(self, documents: list[Document], dimensions: int = 4096,
                 feature_mode: str = "hybrid", min_score: float = 0.18):
        if not 0 <= min_score <= 1 or not np.isfinite(min_score):
            raise ValueError("min_score must be between zero and one")
        if not isinstance(dimensions, int) or not 16 <= dimensions <= 262144:
            raise ValueError("dimensions must be an integer between 16 and 262144")
        if feature_mode not in {"raw_bytes", "words", "hybrid"}:
            raise ValueError("Invalid feature mode")
        self.dimensions = dimensions
        self.feature_mode = feature_mode
        self.min_score = float(min_score)
        self.documents = list(documents)
        if any(not isinstance(d, Document) or not d.id or not isinstance(d.text, str) for d in self.documents):
            raise ValueError("Each document needs a nonempty ID and text")
        if len({d.id for d in self.documents}) != len(self.documents):
            raise ValueError("Document IDs must be unique")
        self._rebuild()

    def _rebuild(self) -> None:
        self.payloads = [encode_text(d.text) for d in self.documents]
        self.spectra = np.array([text_spectrum(d.text, self.dimensions, self.feature_mode)
                                 for d in self.documents], dtype=np.complex128).reshape(-1, self.dimensions)
        self._words = [set(tokenize(d.text)) for d in self.documents]
        serialized = json.dumps([asdict(d) for d in self.documents], ensure_ascii=False, separators=(",", ":"))
        self.archive = encode_text(serialized)

    def scores(self, prompt: str, time_s: float = 0.0) -> np.ndarray:
        query = text_spectrum(prompt, self.dimensions, self.feature_mode)
        phase = np.exp(1j * phase_angles(self.dimensions, time_s))
        # Readout of the coherent cross term. Probe and memory share a clock.
        # This is algebraically half the energy change of K_i + Q; evaluate the
        # cross term directly to avoid catastrophic cancellation near zero.
        keys_t = self.spectra * phase
        query_t = query * phase
        return np.real(np.sum(np.conj(keys_t) * query_t, axis=1))

    def ask(self, prompt: str, top_k: int = 1, time_s: float = 0.0) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must contain text")
        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        scores = self.scores(prompt, time_s)
        query_words = set(tokenize(prompt))
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
            if self.feature_mode != "raw_bytes" and not (query_words & self._words[index]):
                continue
            doc = self.documents[index]
            text = decode_text(self.payloads[index], time_s)
            matches.append({"id": doc.id, "text": text, "source": doc.source,
                            "score": score, "interference": 2.0 * score})
            if len(matches) >= top_k:
                break
        return {
            "answer": "\n\n".join(m["text"] for m in matches) if matches else
                      "Keine ausreichend passende gespeicherte Textstelle gefunden.",
            "abstained": not matches,
            "matches": matches,
            "method": "coherent_fourier_interference_then_inverse_standing_wave",
            "feature_mode": self.feature_mode,
            "time_s": float(time_s),
            "note": "Extraktive Antwort aus dem Wellenspeicher; keine frei gelernte Sprachgenerierung.",
        }

    def snapshot(self, time_s: float, points: int = 256) -> dict:
        result = sample_wave(self.archive, time_s, points)
        # The addressed key bank also evolves, not just the display payload.
        key_phase = np.exp(1j * phase_angles(self.dimensions, time_s))
        keys_t = self.spectra * key_phase
        result.update({"document_count": len(self.documents), "dimensions": self.dimensions,
                       "key_mode_count": int(self.spectra.size),
                       "key_energy": float(np.sum(np.abs(keys_t)**2)),
                       "stored_bytes": self.archive.byte_length,
                       "feature_mode": self.feature_mode})
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
        replacement = WaveMemory(self.documents + [Document(document_id, text, str(source))],
                                 self.dimensions, self.feature_mode, self.min_score)
        self.__dict__.update(replacement.__dict__)
        return document_id

    def save(self, path: str | Path) -> None:
        """Atomic, pickle-free file; corpus texts are stored as wave coefficients."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"format": "freqai-standing-v1", "byte_length": self.archive.byte_length,
                  "sha256": self.archive.sha256, "dimensions": self.dimensions,
                  "feature_mode": self.feature_mode, "min_score": self.min_score}
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
                   float(header["min_score"]))

    def direct_cosine_scores(self, prompt: str) -> np.ndarray:
        """Reference baseline: Parseval requires agreement with wave scores."""
        query = feature_vector(prompt, self.dimensions, self.feature_mode)
        return np.array([np.dot(feature_vector(d.text, self.dimensions, self.feature_mode), query)
                         for d in self.documents])
