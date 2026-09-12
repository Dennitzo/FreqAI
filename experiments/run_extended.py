"""Auditable 120-document extension; development and frozen holdout are separate.

Commands (each writes JSON/CSV):
  python experiments/run_extended.py --stage baseline
  python experiments/run_extended.py --stage development
  python experiments/run_extended.py --stage final --freeze

The original source is archived, so the old implementation remains runnable.
Evaluation never modifies the central database or the running HTTP service.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import csv
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from freqai.codec import decode_text, modal_state, recover_coefficients
from freqai.features import tokenize
from freqai.memory import Document, WaveMemory

OUTPUT = ROOT / "results" / "extended"
EXTENSION = ROOT / "memory" / "fixtures" / "extension_120.jsonl"
QUERIES = ROOT / "memory" / "evaluation" / "extension_queries.json"
CONFIG = {"dimensions": 4096, "feature_mode": "morphology", "min_score": .18,
          "retrieval_policy": "coverage", "min_coverage": .6}
LEGACY_CONFIG = {"dimensions": 4096, "feature_mode": "hybrid", "min_score": .18}


def fixture_path(name: str) -> Path:
    preferred = ROOT / "memory" / "fixtures" / name
    return preferred if preferred.exists() else ROOT / "data" / name


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def load_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    # The active corpus now consists entirely of everyday conversation pairs.
    base = []
    extension = read_jsonl(EXTENSION)
    cases = json.loads(QUERIES.read_text(encoding="utf-8"))["cases"]
    validate_inputs(base, extension, cases)
    return base, extension, cases


def validate_inputs(base: list[dict], extension: list[dict], cases: list[dict]) -> None:
    documents = {item["id"]: item for item in base + extension}
    assert len(extension) >= 120, "At least 120 additional documents required"
    assert len(documents) == len(base) + len(extension), "Duplicate ID"
    assert len({item["text"] for item in documents.values()}) == len(documents), "Duplicate text"
    assert len({item["source"] for item in extension}) >= 10, "Insufficient topic diversity"
    assert Counter(case["split"] for case in cases)["holdout"] >= 60
    assert len({case["id"] for case in cases}) == len(cases)
    texts = {item["text"] for item in documents.values()}
    prompts = {item.get("prompt") for item in documents.values() if item.get("prompt")}
    for case in cases:
        assert case["split"] in {"development", "holdout", "stress"}
        assert case["prompt"] not in texts, "Evaluation prompt leaked into memory"
        assert case["prompt"] not in prompts, "Evaluation phrasing exactly duplicates a stored key"
        expected = case["expected_id"]
        assert expected is None or expected in documents
        if case["group"] == "semantic_no_overlap":
            expected_key = documents[expected].get("prompt") or documents[expected]["text"]
            overlap = set(tokenize(case["prompt"])) & set(tokenize(expected_key))
            assert not overlap, f"Semantic stress has lexical overlap: {case['id']}: {overlap}"


def archived_module():
    name = "freqai_archived_baseline"
    if name not in sys.modules:
        folder = OUTPUT / "baseline_source" / "freqai"
        spec = importlib.util.spec_from_file_location(name, folder / "__init__.py",
                                                    submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def build(items: list[dict], config: dict, legacy: bool = False):
    module = archived_module() if legacy else sys.modules["freqai.memory"]
    if not legacy:
        return module.WaveMemory([module.Document(**item) for item in items], **config)
    # Fair retrieval baseline: retain the original algorithm and answer waves,
    # explicitly supply the same stored conversation prompts as key material.
    # The archived pre-pair implementation itself remains unchanged on disk.
    features = sys.modules["freqai_archived_baseline.features"]
    keys = [item.get("prompt") or item["text"] for item in items]

    class PairedLegacyMemory(module.WaveMemory):
        def _rebuild(self):
            self.payloads = [module.encode_text(doc.text) for doc in self.documents]
            self.spectra = np.asarray([features.text_spectrum(key, self.dimensions, self.feature_mode)
                                      for key in keys], dtype=np.complex128).reshape(-1, self.dimensions)
            self._words = [set(features.tokenize(key)) for key in keys]
            # Same full-archive rebuild, now retaining prompt metadata as well.
            self.archive = module.encode_text(json.dumps(items, ensure_ascii=False, separators=(",", ":")))

    return PairedLegacyMemory([module.Document(**{k: v for k, v in item.items() if k != "prompt"})
                              for item in items], **config)


def provenance() -> dict:
    paths = [EXTENSION, QUERIES, fixture_path("demo.jsonl"), fixture_path("benchmark.json"),
             ROOT / "freqai" / "memory.py", ROOT / "freqai" / "features.py",
             ROOT / "freqai" / "codec.py", Path(__file__)]
    paths.extend(sorted((OUTPUT / "baseline_source" / "freqai").glob("*.py")))
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "numpy": np.__version__, "platform": platform.platform(),
            "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths}}


def write_json(name: str, value) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(memory, cases: list[dict], label: str, time_s: float = 17.123) -> list[dict]:
    rows = []
    for case in cases:
        start = time.perf_counter_ns()
        answer = memory.ask(case["prompt"], time_s=time_s)
        query_ms = (time.perf_counter_ns() - start) / 1e6
        predicted = answer["matches"][0]["id"] if answer["matches"] else None
        expected = case["expected_id"]
        rows.append({"iteration": label, **case, "predicted_id": predicted,
                     "correct": predicted == expected, "abstained": answer["abstained"],
                     "wrong_answer": predicted is not None and predicted != expected,
                     "score": answer["matches"][0]["score"] if answer["matches"] else None,
                     "query_ms": query_ms})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({(row["iteration"], row["split"], row["group"]) for row in rows})
    result = []
    for iteration, split, group in keys:
        selected = [row for row in rows if (row["iteration"], row["split"], row["group"]) ==
                    (iteration, split, group)]
        result.append({"iteration": iteration, "split": split, "group": group,
                       "total": len(selected), "correct": sum(row["correct"] for row in selected),
                       "abstained": sum(row["abstained"] for row in selected),
                       "wrong_answers": sum(row["wrong_answer"] for row in selected),
                       "query_median_ms": float(np.median([row["query_ms"] for row in selected]))})
    return result


def run_baseline(base: list[dict], extension: list[dict], cases: list[dict]) -> dict:
    selected = [case for case in cases if case["split"] == "development"]
    memory = build(base + extension, LEGACY_CONFIG, legacy=True)
    rows = evaluate(memory, selected, "01_legacy_development")
    result = {"provenance": provenance(), "configuration": LEGACY_CONFIG,
              "document_count": len(memory.documents), "holdout_evaluated": False,
              "summary": summarize(rows), "rows": rows}
    write_json("01_baseline.json", result)
    write_csv("01_baseline.csv", rows)
    return result


def run_development(base: list[dict], extension: list[dict], cases: list[dict]) -> dict:
    selected = [case for case in cases if case["split"] == "development"]
    memory = build(base + extension, CONFIG)
    rows = evaluate(memory, selected, "02_coverage_development")
    result = {"provenance": provenance(), "configuration": CONFIG,
              "document_count": len(memory.documents), "holdout_evaluated": False,
              "summary": summarize(rows), "rows": rows}
    write_json("02_development.json", result)
    write_csv("02_development.csv", rows)
    return result


def incremental_experiment(base: list[dict], extension: list[dict]) -> tuple[list[dict], dict]:
    current = build(base, CONFIG)
    rows = []
    roundtrips = stable_answers = checked_answers = 0
    maximum_coefficient_error = maximum_score_error = 0.
    full_grid_checks = 0
    from scipy.fft import dct
    from freqai.codec import decode_coefficients, sample_wave

    for number, item in enumerate(extension, 1):
        previous_packets = current.payloads[:]
        previous_spectra = current.spectra.copy()
        start = time.perf_counter_ns()
        legacy = build(base + extension[:number], LEGACY_CONFIG, legacy=True)
        rebuild_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        current = current.with_documents_added([Document(**item)])
        append_ms = (time.perf_counter_ns() - start) / 1e6
        identities_unchanged = all(packet is current.payloads[index]
                                   for index, packet in enumerate(previous_packets))
        spectra_unchanged = bool(np.array_equal(previous_spectra, current.spectra[:-1]))
        start = time.perf_counter_ns()
        response = current.ask(item.get("prompt") or item["text"], time_s=number * .137)
        query_ms = (time.perf_counter_ns() - start) / 1e6
        immediate_retrieval = response["matches"][0]["id"] == item["id"] if response["matches"] else False
        start = time.perf_counter_ns()
        snapshot = current.snapshot(number * .137, points=64)
        snapshot_ms = (time.perf_counter_ns() - start) / 1e6
        all_keys = snapshot["key_mode_count"] == len(current.documents) * current.dimensions
        all_payloads = snapshot["modal_count"] >= sum(len(packet.coefficients) for packet in current.payloads)
        rows.append({"append_number": number, "document_count": len(current.documents),
                     "whole_rebuild_ms": rebuild_ms, "incremental_append_ms": append_ms,
                     "query_ms": query_ms, "snapshot_ms": snapshot_ms,
                     "old_payload_identity_unchanged": identities_unchanged,
                     "old_spectra_unchanged": spectra_unchanged,
                     "immediate_self_retrieval": immediate_retrieval,
                     "all_key_modes": all_keys, "all_payload_modes": all_payloads,
                     "payload_modes": sum(len(packet.coefficients) for packet in current.payloads),
                     "snapshot_modes": snapshot["modal_count"]})
        assert identities_unchanged and spectra_unchanged and immediate_retrieval and all_keys and all_payloads
        if number in {1, 40, 80, 120}:
            reference = build(base + extension[:number], CONFIG)
            prompts = [extension[0]["prompt"], extension[number - 1]["prompt"], "unbekanntes Quarzflügelschloss"]
            for prompt in prompts:
                expected = current.ask(prompt, time_s=0)["matches"]
                for clock in (0., .0137, 1234.5, 1e12):
                    current_response = current.ask(prompt, time_s=clock)["matches"]
                    stable_answers += [m["id"] for m in current_response] == [m["id"] for m in expected]
                    checked_answers += 1
                    error = float(np.max(np.abs(current.scores(prompt, clock) - reference.direct_cosine_scores(prompt))))
                    maximum_score_error = max(maximum_score_error, error)
            for doc, packet in zip(current.documents, current.payloads):
                for clock in (0., .0137, 1234.5, 1e12):
                    assert decode_text(packet, clock) == doc.text
                    roundtrips += 1
                    q, p = modal_state(packet, clock)
                    error = float(np.max(np.abs(recover_coefficients(q, p, clock) - packet.coefficients)))
                    maximum_coefficient_error = max(maximum_coefficient_error, error)
            packet = current.payloads[-1]
            grid = sample_wave(packet, .371, points=max(2, len(packet.coefficients)))
            restored = recover_coefficients(dct(grid["displacement"], norm="ortho"),
                                             dct(grid["quadrature"], norm="ortho"), .371)
            assert decode_coefficients(restored, packet.byte_length, packet.sha256) == current.documents[-1].text
            full_grid_checks += 1
    summary = {"append_count": len(rows), "final_document_count": len(current.documents),
               "whole_rebuild_total_ms": sum(row["whole_rebuild_ms"] for row in rows),
               "incremental_total_ms": sum(row["incremental_append_ms"] for row in rows),
               "incremental_median_ms": float(np.median([row["incremental_append_ms"] for row in rows])),
               "query_median_ms": float(np.median([row["query_ms"] for row in rows])),
               "snapshot_median_ms": float(np.median([row["snapshot_ms"] for row in rows])),
               "text_roundtrips": roundtrips, "stable_answers": stable_answers,
               "checked_answers": checked_answers, "full_grid_roundtrips": full_grid_checks,
               "maximum_coefficient_error": maximum_coefficient_error,
               "maximum_wave_vs_direct_score_error": maximum_score_error,
               "all_append_invariants_passed": True}
    return rows, summary


def run_final(base: list[dict], extension: list[dict], cases: list[dict]) -> dict:
    # This marker is written BEFORE any held-out inference. No tuning in this stage.
    frozen = {"configuration": CONFIG, "provenance": provenance(),
              "policy": "Parameters fixed before holdout inference; report every failure."}
    write_json("freeze.json", frozen)
    selected = [case for case in cases if case["split"] in {"holdout", "stress"}]
    current = build(base + extension, CONFIG)
    baseline = build(base + extension, LEGACY_CONFIG, legacy=True)
    rows = evaluate(current, selected, "03_frozen_coverage")
    rows.extend(evaluate(baseline, selected, "03_frozen_legacy_comparison"))
    old_benchmark = json.loads(fixture_path("benchmark.json").read_text(encoding="utf-8-sig"))
    old_cases = [{**case, "split": "legacy_regression", "group": "legacy_" + case["group"]}
                 for case in old_benchmark["cases"] if case["prompt"].strip()]
    old_current = build(old_benchmark["documents"], CONFIG)
    old_legacy = build(old_benchmark["documents"], LEGACY_CONFIG, legacy=True)
    rows.extend(evaluate(old_current, old_cases, "03_separate_facts_regression_coverage"))
    rows.extend(evaluate(old_legacy, old_cases, "03_separate_facts_regression_legacy"))
    append_rows, append_summary = incremental_experiment(base, extension)
    result = {"freeze": frozen, "document_count": len(current.documents),
              "summary": summarize(rows), "rows": rows, "incremental": append_summary}
    write_json("03_final.json", result)
    write_csv("03_final.csv", rows)
    write_csv("incremental_120.csv", append_rows)
    write_json("incremental_120.json", {"summary": append_summary, "rows": append_rows})
    write_report()
    return result


def write_report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads((OUTPUT / name).read_text(encoding="utf-8"))
               for name in ("01_baseline.json", "02_development.json", "03_final.json")]
    summaries = [row for record in records for row in record["summary"]]
    write_csv("summary.csv", summaries)
    lines = ["# 120 Alltagspaare im zentralen Gesprächsspeicher", "",
             "120 vollständig neue, handgeschriebene Paare aus Gesprächsanlass und Antwort, verteilt auf 12 Alltagsthemen. Der aktive Korpus enthält ausschließlich diese Gesprächspaare.",
             "60 Entwicklungsfragen, 84 zurückgehaltene Fragen und 20 Stressfragen liegen separat und werden nicht als Antworttexte importiert.",
             "Die festen Regeln wurden vor der Holdout-Auswertung eingefroren; keine Modellgewichte und keine Synonymlisten aus den Antworten gelernt.",
             "Die alte Implementierung erhält für den fairen Vergleich dieselben gespeicherten Anlassprompts als FFT-Schlüssel. Ihre Suchregeln und Antwortwellen bleiben unverändert. Der Adapter ist im Experimentcode ausgewiesen.",
             "Der frühere Faktenbenchmark wird zusätzlich isoliert geprüft und gehört weder zum aktiven Gesprächsspeicher noch zu den 120 Dialogpaaren. Die überholte Betriebsnotizen-Runde liegt nur unter superseded/.", "",
             "| Iteration | Split | Gruppe | Richtig | Enthalten | Falsche Antworten |", "|---|---|---|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['iteration']} | {row['split']} | {row['group']} | {row['correct']}/{row['total']} | {row['abstained']} | {row['wrong_answers']} |")
    numerical = records[-1]["incremental"]
    lines.extend(["", "## Inkrementeller numerischer Lauf", "",
                  f"Alle {numerical['append_count']} einzelnen Ergänzungen waren sofort auslesbar; alte Nutzdatenobjekte und alte Schlüsselkoeffizienten blieben unverändert.",
                  f"{numerical['text_roundtrips']} exakte Text-Rückrechnungen bei vier Zeiten bis 1e12 s; {numerical['stable_answers']}/{numerical['checked_answers']} phasenstabile Antworten; {numerical['full_grid_roundtrips']} unabhängige Rückrechnungen aus vollständigen Ortsgittern.",
                  f"Maximaler Koeffizientenfehler: {numerical['maximum_coefficient_error']:.3g}; Differenz Wellenkorrelation gegen direktes Skalarprodukt: {numerical['maximum_wave_vs_direct_score_error']:.3g}.",
                  f"Gesamtdauer für 120 Änderungen: alter Komplettaufbau {numerical['whole_rebuild_total_ms']:.1f} ms, inkrementell {numerical['incremental_total_ms']:.1f} ms.",
                  f"Median: Ergänzung {numerical['incremental_median_ms']:.2f} ms, Abfrage {numerical['query_median_ms']:.2f} ms, vollständige Schwingungsberechnung mit Anzeigeverdichtung {numerical['snapshot_median_ms']:.2f} ms.",
                  "Zeitmessungen sind lokale Einzelmessungen im laufenden Entwicklungssystem, keine universellen Leistungszusagen. Datenbank-I/O und HTTP kommen im separaten Live-Abnahmetest hinzu.", "",
                  "## Interpretation und Grenzen", "",
                  "Die Fourierkorrelation stimmt numerisch mit einem direkten Ähnlichkeitsskalarprodukt überein. Der Nutzen ist verlustfreier Wellenspeicher plus nachvollziehbare Suche, kein Nachweis zusätzlichen Sprachverständnisses durch Schwingungen.",
                  "Abdeckung und feste Wortnormalisierung können falsche Themenantworten reduzieren, zugleich aber belegte Fragen zurückweisen. Die Fehlertabelle enthält auch solche Rückschritte.",
                  "Die bekannten Fragen verwenden weitgehend dieselben Wörter wie die gespeicherten Gesprächsanlässe. Die Stressfragen ohne Wortüberlappung prüfen ausdrücklich die Grenze dieses Verfahrens. Ein Treffer in dieser Gruppe kann zufällig sein und belegt keine allgemeine Semantik.",
                  "Die 120 Antworttexte sind handgeschriebene Gesprächsbeispiele. Die Ergebnisse prüfen das Auswählen passender Antworten, keine freie Textgenerierung und kein über mehrere Gesprächsschritte anhaltendes Kontextverständnis.", "",
                  "![Treffer und falsche Antworten](retrieval.png)", "", "![Ergänzungsdauer](incremental.png)", "",
                  "## Fehler im eingefrorenen Lauf", ""])
    failures = [row for row in records[-1]["rows"] if row["iteration"] == "03_frozen_coverage" and not row["correct"]]
    lines.extend(f"- `{row['id']}` ({row['split']}): {row['prompt']} Erwartet `{row['expected_id']}`, erhalten `{row['predicted_id']}`." for row in failures)
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    chosen = [row for row in summaries if row["split"] in {"development", "holdout"}]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(chosen))
    ax.bar(x - .18, [row["correct"] / row["total"] for row in chosen], .36, label="Correct / total")
    ax.bar(x + .18, [row["wrong_answers"] / row["total"] for row in chosen], .36, label="Wrong answer / total")
    ax.set_xticks(x, [row["iteration"].replace("_", " ") + "\n" + row["group"].replace("_", " ") for row in chosen], rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.set_title("120 everyday conversation pairs: lexical retrieval and abstention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "retrieval.png", dpi=160)
    plt.close(fig)
    appends = json.loads((OUTPUT / "incremental_120.json").read_text(encoding="utf-8"))["rows"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([row["document_count"] for row in appends], [row["whole_rebuild_ms"] for row in appends], label="Original full rebuild")
    ax.plot([row["document_count"] for row in appends], [row["incremental_append_ms"] for row in appends], label="Incremental append")
    ax.set_xlabel("Stored documents")
    ax.set_ylabel("Elapsed milliseconds per addition")
    ax.set_title("Single additions; local timings without database or HTTP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "incremental.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "development", "final"), required=True)
    parser.add_argument("--freeze", action="store_true", help="Explicitly freeze parameters before held-out inference")
    args = parser.parse_args()
    if args.stage == "final" and not args.freeze:
        parser.error("Final evaluation requires --freeze; do not tune on held-out results")
    base, extension, cases = load_inputs()
    function = {"baseline": run_baseline, "development": run_development, "final": run_final}[args.stage]
    result = function(base, extension, cases)
    print(json.dumps({"stage": args.stage, "summary": result["summary"],
                      "incremental": result.get("incremental")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
