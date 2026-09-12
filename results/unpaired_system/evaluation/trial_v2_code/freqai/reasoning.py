"""Explicit is_a inference through a Fourier-basis graph operator.

This is symbolic transitive closure in another numerical basis, not learned
language understanding. Dense storage is O(V**2). Every accepted answer has a
checked path through supplied triples. The German output uses the fixed template
``subject ist ein target`` (appropriate for masculine/neuter class names).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class WaveReasoner:
    """A finite, explicit, reflexive/transitive ``is_a`` graph in wave space.

    Input triples contain subject, predicate='is_a', object, and optional source.
    Labels are exact and case-sensitive; no language parsing is performed.
    Projection to a Boolean frontier after each wave step limits numerical
    growth. This nonlinear readout and visited-state bookkeeping are part of the
    algorithm, rather than an autonomous linear physical oscillation.
    """

    MAX_NODES = 512
    NUMERICAL_TOLERANCE = 1e-8

    def __init__(self, triples: list[dict]):
        if not isinstance(triples, list):
            raise TypeError("triples must be a list of explicit relation dictionaries")
        edges: dict[tuple[str, str], dict] = {}
        for index, supplied in enumerate(triples):
            if not isinstance(supplied, Mapping):
                raise TypeError(f"triple {index} must be a dictionary")
            if supplied.get("predicate") != "is_a":
                raise ValueError(f"triple {index}: only predicate='is_a' is supported")
            for key in ("subject", "object"):
                if not isinstance(supplied.get(key), str) or not supplied[key].strip():
                    raise ValueError(f"triple {index}: {key} must be a nonempty string")
            source = supplied.get("source", f"input:triple:{index}")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"triple {index}: source must be a nonempty string")
            triple = {"subject": supplied["subject"], "predicate": "is_a",
                      "object": supplied["object"], "source": source}
            # One reproducible witness per edge; duplicate edges are Boolean.
            edge = (triple["subject"], triple["object"])
            if edge not in edges or source < edges[edge]["source"]:
                edges[edge] = triple
        self.nodes = tuple(sorted({node for edge in edges for node in edge}))
        self.node_index = {node: index for index, node in enumerate(self.nodes)}
        if len(self.nodes) > self.MAX_NODES:
            raise ValueError(f"dense Fourier reasoner is limited to {self.MAX_NODES} nodes")
        self._edges = edges
        self.triples = tuple(dict(edges[edge]) for edge in sorted(edges))
        count = len(self.nodes)
        adjacency = np.zeros((count, count), dtype=np.float64)
        for subject, target in edges:
            adjacency[self.node_index[target], self.node_index[subject]] = 1.0
        self.adjacency = adjacency
        if count:
            fourier = np.fft.fft(np.eye(count), axis=0, norm="ortho")
            self.operator = fourier @ adjacency @ fourier.conj().T
            self.operator_error = float(np.max(np.abs(
                fourier.conj().T @ self.operator @ fourier - adjacency)))
        else:
            self.operator = np.zeros((0, 0), dtype=np.complex128)
            self.operator_error = 0.0
        self.adjacency.flags.writeable = False
        self.operator.flags.writeable = False

    @staticmethod
    def _label(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty exact label")

    def infer(self, subject: str, target: str, max_hops: int = 8) -> dict:
        """Prove a known is_a relation or return an explicit unknown result.

        ``entailed=False`` means unproved, never a proof of negation. Identity
        for a known node follows the explicitly declared reflexivity rule.
        ``steps`` is the number of frequency-space matrix/vector products.
        """
        self._label(subject, "subject")
        self._label(target, "target")
        if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 0:
            raise ValueError("max_hops must be a nonnegative integer")
        result = {
            "subject": subject, "target": target, "predicate": "is_a",
            "answer": f"Ob {subject} ein {target} ist, bleibt anhand der gespeicherten Beziehungen unbekannt.",
            "entailed": False, "status": "unknown", "path": [], "proof": [],
            "sources": [], "steps": 0, "max_hops": max_hops,
            "reason": "unknown_node", "node_count": len(self.nodes),
            "numerical_tolerance": self.NUMERICAL_TOLERANCE,
            "max_imaginary_error": 0.0, "max_integer_error": 0.0,
            "operator_reconstruction_error": self.operator_error,
            "rules": ["explicit is_a edges", "is_a transitivity", "known-node is_a reflexivity"],
            "method": "H=F A F*, repeated H@v, inverse FFT, Boolean frontier projection",
            "general_language_model": False,
        }
        if subject not in self.node_index or target not in self.node_index:
            return result
        start, goal = self.node_index[subject], self.node_index[target]
        if start == goal:
            return self._proved(result, [subject], "reflexivity")
        if max_hops == 0:
            result["reason"] = "hop_limit"
            return result
        visited = np.zeros(len(self.nodes), dtype=bool)
        frontier = np.zeros(len(self.nodes), dtype=bool)
        visited[start] = frontier[start] = True
        predecessor: dict[int, int] = {}
        wave = np.fft.fft(frontier.astype(np.float64), norm="ortho")
        for step in range(1, min(max_hops, len(self.nodes)) + 1):
            propagated = self.operator @ wave
            readout = np.fft.ifft(propagated, norm="ortho")
            result["steps"] = step
            if not np.isfinite(readout).all():
                result["reason"] = "numerical_error"
                return result
            rounded = np.rint(readout.real)
            imaginary_error = float(np.max(np.abs(readout.imag)))
            integer_error = float(np.max(np.abs(readout.real - rounded)))
            result["max_imaginary_error"] = max(result["max_imaginary_error"], imaginary_error)
            result["max_integer_error"] = max(result["max_integer_error"], integer_error)
            if (max(imaginary_error, integer_error) > self.NUMERICAL_TOLERANCE
                    or np.any(rounded < 0) or np.any(rounded > frontier.sum())):
                result["reason"] = "numerical_error"
                return result
            following = (rounded >= 1) & ~visited
            # Every newly asserted node needs a supplied, checkable witness.
            # Numerical propagation selects candidates; explicit edges justify
            # them and prevent a corrupt operator from fabricating relations.
            for child in np.flatnonzero(following):
                parents = [parent for parent in np.flatnonzero(frontier)
                           if (self.nodes[parent], self.nodes[child]) in self._edges]
                if not parents:
                    result["reason"] = "numerical_error"
                    return result
                predecessor[int(child)] = int(parents[0])
            if following[goal]:
                path_indexes = [goal]
                while path_indexes[-1] != start:
                    path_indexes.append(predecessor[path_indexes[-1]])
                path = [self.nodes[index] for index in reversed(path_indexes)]
                return self._proved(result, path, "proved_by_transitivity")
            if not following.any():
                result["reason"] = "no_path"
                return result
            visited |= following
            frontier = following
            wave = np.fft.fft(frontier.astype(np.float64), norm="ortho")
        result["reason"] = "hop_limit"
        return result

    def _proved(self, result: dict, path: list[str], reason: str) -> dict:
        proof = [dict(self._edges[(left, right)]) for left, right in zip(path, path[1:])]
        result.update({"answer": f"{result['subject']} ist ein {result['target']}.",
                       "entailed": True, "status": "proved", "path": path,
                       "proof": proof, "sources": list(dict.fromkeys(edge["source"] for edge in proof)),
                       "reason": reason})
        return result
