"""Reproducible bounded comparison of wave inference and ordinary graph BFS.

Run from any directory with the project's Python interpreter. This experiment
does not tune parameters or change the supplied knowledge graph.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freqai.reasoning import WaveReasoner  # noqa: E402


def graph_reference(edges: list[dict], subject: str, target: str, max_hops: int) -> tuple[bool, int | None]:
    """Independent discrete BFS, with the same explicit reflexivity convention."""
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge["subject"], set()).add(edge["object"])
        adjacency.setdefault(edge["object"], set())
    if subject not in adjacency or target not in adjacency:
        return False, None
    seen = {subject}
    queue = deque([(subject, 0)])
    while queue:
        current, depth = queue.popleft()
        if current == target:
            return True, depth
        if depth >= max_hops:
            continue
        for following in sorted(adjacency[current]):
            if following not in seen:
                seen.add(following)
                queue.append((following, depth + 1))
    return False, None


def run(output_dir: Path, seed: int = 20260906, graph_count: int = 96) -> dict:
    if graph_count < 13 or graph_count > 128:
        raise ValueError("graph_count must be between 13 and 128 (104 to 1024 query cases)")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    cases, graphs = [], []
    for graph_id in range(graph_count):
        node_count = [2, 4, 8, 16, 32, 64][graph_id % 6]
        probability = [0.02, 0.08, 0.25][(graph_id // 6) % 3]
        adjacency = rng.random((node_count, node_count)) < probability
        # Explicit identities register even isolated nodes in the triple schema.
        np.fill_diagonal(adjacency, True)
        edges = [{"subject": f"n{left}", "predicate": "is_a", "object": f"n{right}",
                  "source": f"synthetic:graph:{graph_id}:edge:{left}:{right}"}
                 for left, right in zip(*np.nonzero(adjacency))]
        graphs.append({"graph_id": graph_id, "node_count": node_count,
                       "edge_probability": probability, "triples": edges})
        reasoner = WaveReasoner(edges)
        for query_id in range(8):
            subject, target = (f"n{int(rng.integers(node_count))}" for _ in range(2))
            if query_id == 0:
                target = "unknown_node"
            max_hops = [0, 1, 2, 4, 8, 64, 3, 6][query_id]
            expected, minimum_steps = graph_reference(edges, subject, target, max_hops)
            answer = reasoner.infer(subject, target, max_hops=max_hops)
            proof_valid = all(edge in edges for edge in answer["proof"])
            shortest_path = not expected or answer["steps"] == minimum_steps
            cases.append({"graph_id": graph_id, "query_id": query_id,
                          "subject": subject, "target": target, "max_hops": max_hops,
                          "expected_entailed": expected, "reference_steps": minimum_steps,
                          "actual": answer, "proof_valid": proof_valid,
                          "shortest_path": shortest_path,
                          "pass": answer["entailed"] == expected and proof_valid and shortest_path
                          and answer["reason"] != "numerical_error"})
    supplied = json.loads((ROOT / "memory" / "fixtures" / "relations.json").read_text(encoding="utf-8"))
    demo = WaveReasoner(supplied)
    examples = {
        "new_two_hop_answer": demo.infer("Pudel", "Tier"),
        "new_three_hop_answer": demo.infer("Pudel", "Lebewesen"),
        "hop_limit": demo.infer("Pudel", "Tier", max_hops=1),
        "reverse_relation_is_unknown": demo.infer("Tier", "Pudel"),
        "unrelated_classes_are_unknown": demo.infer("Pudel", "Baum"),
        "unknown_entity": demo.infer("Einhorn", "Tier"),
    }
    new_answer = examples["new_two_hop_answer"]
    direct_edge_exists = any(edge["subject"] == "Pudel" and edge["object"] == "Tier"
                             for edge in supplied)
    all_pass = all(case["pass"] for case in cases)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "graph_count": graph_count, "query_count": len(cases),
        "node_counts": [2, 4, 8, 16, 32, 64], "edge_probabilities": [0.02, 0.08, 0.25],
        "passes": sum(case["pass"] for case in cases),
        "failures": sum(not case["pass"] for case in cases),
        "positive_cases": sum(case["expected_entailed"] for case in cases),
        "negative_or_unknown_cases": sum(not case["expected_entailed"] for case in cases),
        "numerical_failures": sum(case["actual"]["reason"] == "numerical_error" for case in cases),
        "max_integer_error": max(case["actual"]["max_integer_error"] for case in cases),
        "max_imaginary_error": max(case["actual"]["max_imaginary_error"] for case in cases),
        "max_operator_reconstruction_error": max(case["actual"]["operator_reconstruction_error"] for case in cases),
        "numerical_tolerance": WaveReasoner.NUMERICAL_TOLERANCE,
        "new_answer_entailed": new_answer["entailed"],
        "new_answer_direct_edge_absent": not direct_edge_exists,
        "new_answer": new_answer["answer"],
        "passed": all_pass and new_answer["entailed"] and not direct_edge_exists,
        "elapsed_seconds": time.perf_counter() - started,
        "limits": [
            "Explicit supplied is_a semantics, reflexivity and transitivity only; no general NLP",
            "German answer uses a fixed masculine/neuter noun template, without learned grammar",
            "Dense Fourier operator: O(V^2) storage and work per propagation step",
            "Boolean projection and visited-state bookkeeping required between wave-space steps",
            "Fourier basis changes representation; inference power equals ordinary graph reachability",
            "No closed-world negation: unproved relations remain unknown",
            "Synthetic graph agreement does not establish open-domain language reasoning",
        ],
    }
    for filename, value in [("summary.json", summary), ("examples.json", examples), ("graphs.json", graphs)]:
        (output_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
    report = f"""# Fourier-Inferenz: numerischer Bericht

Zeitpunkt: {summary['created_at_utc']}; Seed: {seed}.

**{summary['passes']}/{len(cases)} Fälle bestanden**, {summary['failures']} Fehler.
Verglichen wurden {graph_count} zufällige Graphen mit 2 bis 64 Knoten gegen eine
unabhängige klassische Breitensuche. Enthalten sind Identitäten, gerichtete
Zyklen, unbekannte Knoten, nicht ableitbare Beziehungen und beschränkte Pfadlängen.
{summary['positive_cases']} Fälle waren ableitbar und
{summary['negative_or_unknown_cases']} waren unbelegt beziehungsweise unbekannt.

Der Operator ist H = F A F*, mit A[object, subject] = 1. Jeder Schritt berechnet
H @ v im Frequenzraum. Die inverse FFT liefert ganzzahlige Kantenzählungen;
eine kontrollierte boolesche Projektion bildet die nächste Front. Ein Beweispfad
aus den tatsächlichen Eingangstripeln begründet jede ausgegebene Behauptung.

Größter Ganzzahlfehler: {summary['max_integer_error']:.3e}; größter imaginärer
Rest: {summary['max_imaginary_error']:.3e}; Toleranz: {WaveReasoner.NUMERICAL_TOLERANCE:.1e}.
Numerische Fehlerfälle: {summary['numerical_failures']}.
Gesamtrechenzeit: {summary['elapsed_seconds']:.3f} Sekunden.

Neue berechnete Antwort: **{new_answer['answer']}**
Beweis: {' → '.join(new_answer['path'])}.
Das direkte Tripel Pudel → Tier ist in den Eingangsdaten nicht enthalten.
Quellen: {', '.join(new_answer['sources'])}. Die Quellen sind explizite lokale
Demofakten; ihre Kennungen sind keine externe wissenschaftliche Validierung.

Die Methode hat dieselbe Folgerungsfähigkeit wie Graph-Reichbarkeit. Die
Fouriertransformation allein erzeugt keine neue Semantik. Für V Knoten benötigt
der dichte Operator O(V²) Speicher und Rechenarbeit je Schritt. Semantik und
Antwortschablone sind vorgegeben. Die Lösung demonstriert eine neue, begründete
Antwort aus gespeicherten Beziehungen, keine allgemeine KI.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "reasoning")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--graphs", type=int, default=96)
    options = parser.parse_args()
    result = run(options.output, seed=options.seed, graph_count=options.graphs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
