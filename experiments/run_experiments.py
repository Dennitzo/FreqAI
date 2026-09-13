"""Reproducible, non-training numerical experiments for the FreqAI prototype.

Run: .venv/Scripts/python.exe experiments/run_experiments.py --benchmark PATH.json
All benchmark prompts remain outside the memory. Failures are reported, not
silently excluded. Configuration comparisons are diagnostic, not an independent
held-out evaluation of the selected configuration.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import dct, fft
from scipy.special import erf

from freqai.codec import (
    WavePacket, decode_coefficients, decode_text, encode_text, modal_state,
    phase_angles, recover_coefficients, sample_wave,
)
from freqai.features import feature_vector
from freqai.memory import Document, WaveMemory


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_benchmark(benchmark: dict) -> None:
    documents = {doc["id"]: doc["text"] for doc in benchmark["documents"]}
    if len(documents) != len(benchmark["documents"]):
        raise ValueError("Duplicate benchmark document IDs")
    groups = {name: [case for case in benchmark["cases"] if case["group"] == name]
              for name in ("known", "null", "semantic_no_overlap")}
    if not (len(groups["known"]) >= 30 and len(groups["null"]) >= 10
            and len(groups["semantic_no_overlap"]) >= 8):
        raise ValueError("Benchmark does not meet minimum group sizes")
    for case in benchmark["cases"]:
        if case["prompt"] in documents.values():
            raise ValueError("Evaluation prompt leaked into memory")
        if case["expected_id"] is not None and case["expected_id"] not in documents:
            raise ValueError("Unknown expected document")
        if case["group"] == "semantic_no_overlap":
            words = lambda text: set(re.findall(r"[^\W_]+", text.casefold()))
            if words(case["prompt"]) & words(documents[case["expected_id"]]):
                raise ValueError(f"Paraphrase has word overlap: {case['id']}")


def codec_experiments() -> tuple[list[dict], list[dict], list[dict]]:
    texts = ["", "A", "\x00\x01\x7f", "Hallo, stehende Welle!",
             "Grüße: äöü ÄÖÜ ß €", "中文、日本語、한국어", "مرحبا بالعالم",
             "🌊🧠👩🏽‍🔬 e\u0301 é", "Zeile 1\nZeile 2\tEnde",
             "Fourier speichert Reihenfolge, aber keine Bedeutung. " * 100]
    rng = np.random.default_rng(20260906)
    alphabet = np.array(list("abcXYZ 0123äüß中文🌊"))
    texts.extend("".join(rng.choice(alphabet, size=n)) for n in (17, 127, 513))
    times = [0.0, 1e-6, 0.0137, 0.25, 1.0, 17.123, 1e3, 1e6, 1e12]
    rows = []
    for index, text in enumerate(texts):
        packet = encode_text(text)
        energy0 = float(np.dot(packet.coefficients, packet.coefficients))
        spatial = (np.frombuffer(text.encode("utf-8") or b"\x00", dtype=np.uint8)
                   .astype(float) - 127.5) / 127.5
        parseval_error = abs(energy0 - float(spatial @ spatial)) / max(1., energy0)
        omega = 2 * np.pi * np.arange(len(packet.coefficients)) * 30 / len(packet.coefficients)
        physical0 = float(0.5 * np.sum(omega**2 * packet.coefficients**2))
        for current_time in times:
            q, p = modal_state(packet, current_time)
            recovered = recover_coefficients(q, p, current_time)
            snapshot = sample_wave(packet, current_time, points=max(2, len(q)))
            full_q = dct(snapshot["displacement"], norm="ortho")
            full_p = dct(snapshot["quadrature"], norm="ortho")
            decoded_full = decode_coefficients(recover_coefficients(full_q, full_p, current_time),
                                               packet.byte_length, packet.sha256)
            rows.append({
                "text_id": index, "utf8_bytes": packet.byte_length, "modes": len(q),
                "time_s": current_time, "exact_text": decode_text(packet, current_time) == text,
                "full_grid_exact_text": decoded_full == text,
                "coefficient_max_error": float(np.max(np.abs(recovered - packet.coefficients))),
                "relative_energy_error": abs(float(q @ q + p @ p) - energy0) / max(1., energy0),
                "relative_physical_energy_error": abs(snapshot["physical_energy"] - physical0) / max(1., physical0),
                "parseval_relative_error": parseval_error,
                "coefficient_bytes": int(packet.coefficients.nbytes),
            })
    packet = encode_text("Unicode-Wellen: Grüße 中文 🌊. Jede Position bleibt erhalten. " * 4)
    noise_rows = []
    for sigma in (0., 1e-6, 1e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2):
        for seed in range(32):
            perturbation = np.random.default_rng(seed).normal(0., sigma, size=len(packet.coefficients))
            noisy = WavePacket(packet.coefficients + perturbation, packet.byte_length, packet.sha256)
            try:
                exact = decode_text(noisy, time_s=123.456) == decode_text(packet)
                rejected, message = False, ""
            except (ValueError, UnicodeDecodeError) as exc:
                exact, rejected, message = False, True, str(exc)
            per_byte = 1. if sigma == 0 else float(erf(.5 / (127.5 * sigma * math.sqrt(2))))
            noise_rows.append({"sigma_coefficients": sigma, "seed": seed, "bytes": packet.byte_length,
                               "exact_text": exact, "corruption_rejected": rejected,
                               "undetected_wrong_text": not exact and not rejected,
                               "ideal_gaussian_success_probability": per_byte**packet.byte_length,
                               "error": message})
    # A displacement sample alone can hide a mode completely; quadrature fixes it.
    position_rows = []
    for size in (16, 64, 256):
        coefficients = np.zeros(size)
        coefficients[1] = 1.
        packet = WavePacket(coefficients, size, "not-a-text-demonstration")
        current_time = size / 120.  # omega_1 * t = pi / 2.
        q, p = modal_state(packet, current_time)
        position_rows.append({"modes": size, "time_s": current_time,
                              "displacement_l2": float(np.linalg.norm(q)),
                              "quadrature_l2": float(np.linalg.norm(p)),
                              "recovered_coefficient_error": float(np.max(np.abs(
                                  recover_coefficients(q, p, current_time) - coefficients))),
                              "position_only_mode_invisible": float(np.linalg.norm(q)) < 1e-12})
    return rows, noise_rows, position_rows


def retrieval_experiments(benchmark: dict) -> tuple[list[dict], list[dict], list[dict]]:
    documents = [Document(**doc) for doc in benchmark["documents"]]
    doc_ids = [doc.id for doc in documents]
    rows, summaries, invariants = [], [], []
    for mode in ("raw_bytes", "words", "hybrid"):
        for dimensions in (256, 1024, 4096, 16384):
            started = time.perf_counter()
            memory = WaveMemory(documents, dimensions=dimensions, feature_mode=mode, min_score=.18)
            for case in benchmark["cases"]:
                scores = memory.scores(case["prompt"])
                best = int(np.argmax(scores))
                error = ""
                input_rejected = False
                try:
                    result = memory.ask(case["prompt"])
                except ValueError as exc:
                    if case["prompt"].strip():
                        raise
                    input_rejected, error = True, str(exc)
                    result = {"matches": [], "abstained": True, "answer": ""}
                selected = result["matches"][0]["id"] if result["matches"] else None
                expected = case["expected_id"]
                rows.append({
                    "feature_mode": mode, "dimensions": dimensions, "case_id": case["id"],
                    "group": case["group"], "prompt": case["prompt"], "expected_id": expected,
                    "selected_id": selected, "ranked_first_id": doc_ids[best],
                    "correct": selected == expected,
                    "ranked_correct": expected is not None and doc_ids[best] == expected,
                    "abstained": bool(result["abstained"]), "input_rejected": input_rejected,
                    "top_score": float(scores[best]),
                    "expected_score": float(scores[doc_ids.index(expected)]) if expected else None,
                    "answer": result["answer"], "error": error,
                })
            subset = [row for row in rows if row["feature_mode"] == mode and row["dimensions"] == dimensions]
            for group in ("known", "null", "semantic_no_overlap"):
                grouped = [row for row in subset if row["group"] == group]
                summaries.append({"feature_mode": mode, "dimensions": dimensions, "group": group,
                                  "cases": len(grouped), "correct": sum(row["correct"] for row in grouped),
                                  "accuracy": float(np.mean([row["correct"] for row in grouped])),
                                  "ranked_accuracy": float(np.mean([row["ranked_correct"] for row in grouped])) if group != "null" else None,
                                  "abstained": sum(row["abstained"] for row in grouped),
                                  "input_rejected": sum(row["input_rejected"] for row in grouped)})
            for case in benchmark["cases"][:3]:
                baseline = memory.direct_cosine_scores(case["prompt"])
                initial = memory.ask(case["prompt"], time_s=0.)
                query = fft(feature_vector(case["prompt"], dimensions, mode), norm="ortho")
                explicit_energy_cross_term = .5 * (np.sum(np.abs(memory.spectra + query)**2, axis=1)
                    - np.sum(np.abs(memory.spectra)**2, axis=1) - np.sum(np.abs(query)**2))
                for current_time in (0., .0137, 1., 1234.5, 1e12):
                    scores = memory.scores(case["prompt"], time_s=current_time)
                    snapshot = memory.snapshot(current_time, points=64)
                    invariants.append({
                        "feature_mode": mode, "dimensions": dimensions, "case_id": case["id"],
                        "time_s": current_time,
                        "direct_cosine_max_error": float(np.max(np.abs(scores - baseline))),
                        "explicit_interference_max_error": float(np.max(np.abs(scores - explicit_energy_cross_term))),
                        "answer_invariant": memory.ask(case["prompt"], time_s=current_time)["answer"] == initial["answer"],
                        "key_energy": snapshot["key_energy"],
                        "key_energy_error": abs(snapshot["key_energy"] - float(np.sum(np.abs(memory.spectra)**2))),
                    })
            known = next(row for row in summaries if row["feature_mode"] == mode and row["dimensions"] == dimensions and row["group"] == "known")
            print(f"Retrieval {mode:9s} D={dimensions:5d}: {known['correct']}/{known['cases']} known, {time.perf_counter()-started:.2f}s", flush=True)
    return rows, summaries, invariants


def hrr_capacity() -> list[dict]:
    """Separate HRR simulation: not the addressed, exact main memory.

    Random unitary keys bind independent random unit vectors. All bindings are
    summed into one spectrum. Unbinding is noisy; a clean-up dictionary remains
    necessary for identification. Multiple fixed seeds expose cross-talk.
    """
    rows = []
    for dimensions in (256, 1024, 4096):
        for count in (8, 32, 128, 512):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                values = rng.normal(size=(count, dimensions))
                values /= np.linalg.norm(values, axis=1, keepdims=True)
                phases = rng.uniform(0, 2 * np.pi, size=(count, dimensions // 2 + 1))
                keys = np.exp(1j * phases)
                keys[:, 0] = rng.choice([-1., 1.], size=count)
                keys[:, -1] = rng.choice([-1., 1.], size=count)
                spectrum = np.sum(keys * np.fft.rfft(values, norm="ortho", axis=1), axis=0)
                queried = rng.choice(count, size=min(16, count), replace=False)
                retrieved = np.fft.irfft(np.conj(keys[queried]) * spectrum, n=dimensions, norm="ortho", axis=1)
                norms = np.linalg.norm(retrieved, axis=1)
                candidates = (retrieved @ values.T) / norms[:, None]
                correct = np.argmax(candidates, axis=1) == queried
                wanted = candidates[np.arange(len(queried)), queried]
                residual = retrieved - values[queried]
                rows.append({"dimensions": dimensions, "stored_items": count, "seed": seed,
                             "queries": len(queried), "correct": int(np.sum(correct)),
                             "accuracy": float(np.mean(correct)),
                             "mean_target_cosine": float(np.mean(wanted)),
                             "mean_noise_energy": float(np.mean(np.sum(residual**2, axis=1))),
                             "predicted_noise_energy": count - 1,
                             "predicted_target_cosine_approx": 1 / math.sqrt(count),
                             "cleanup_dictionary_required": True})
        print(f"HRR superposition D={dimensions}: complete", flush=True)
    return rows


def hash_collisions() -> list[dict]:
    rows = []
    count = 2000
    for dimensions in (256, 1024, 4096, 16384):
        indices = [int(np.argmax(np.abs(feature_vector(f"zzsynthetic{index}token", dimensions, "words"))))
                   for index in range(count)]
        histogram = np.bincount(indices, minlength=dimensions)
        occupied = int(np.count_nonzero(histogram))
        rows.append({"dimensions": dimensions, "unique_input_tokens": count,
                     "occupied_buckets": occupied, "bucket_collisions": count - occupied,
                     "maximum_bucket_load": int(histogram.max()),
                     "expected_collisions_uniform_hash": count - dimensions * (1 - (1 - 1 / dimensions)**count)})
    return rows


def plots(destination: Path, summaries: list[dict], noise: list[dict], capacity: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, group, title in zip(axes, ("known", "null", "semantic_no_overlap"),
                                 ("Bekannte Fragen (32)", "Nullfälle (10)", "Paraphrasen ohne Wortüberlappung (8)")):
        for mode in ("raw_bytes", "words", "hybrid"):
            subset = [row for row in summaries if row["feature_mode"] == mode and row["group"] == group]
            axis.plot([row["dimensions"] for row in subset], [row["accuracy"] for row in subset], marker="o", label=mode)
        axis.set_xscale("log", base=2)
        axis.set_title(title)
        axis.set_xlabel("Fourier-Dimension D")
        axis.set_ylim(-.04, 1.04)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Anteil korrekter Systemantworten")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(destination / "retrieval_accuracy.png")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    sigmas = sorted({row["sigma_coefficients"] for row in noise if row["sigma_coefficients"] > 0})
    rates = [np.mean([row["exact_text"] for row in noise if row["sigma_coefficients"] == sigma]) for sigma in sigmas]
    theoretical = [next(row["ideal_gaussian_success_probability"] for row in noise if row["sigma_coefficients"] == sigma) for sigma in sigmas]
    axis.semilogx(sigmas, rates, "o-", label="Gemessen, 32 feste Seeds")
    axis.semilogx(sigmas, theoretical, "--", label="Gauß-Modell, unabhängige Bytes")
    axis.set(xlabel="Standardabweichung des Koeffizientenrauschens", ylabel="Exakte Textrekonstruktion", ylim=(-.04, 1.04))
    axis.legend()
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(destination / "noise_tolerance.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for dimensions in (256, 1024, 4096):
        counts = (8, 32, 128, 512)
        means = [np.mean([row["accuracy"] for row in capacity if row["dimensions"] == dimensions and row["stored_items"] == count]) for count in counts]
        axes[0].semilogx(counts, means, "o-", label=f"D={dimensions}")
        noise_energies = [np.mean([row["mean_noise_energy"] for row in capacity if row["dimensions"] == dimensions and row["stored_items"] == count]) for count in counts]
        axes[1].loglog(counts, noise_energies, "o-", label=f"D={dimensions}")
    axes[1].loglog(counts, np.array(counts) - 1, "k--", label="Erwartung N-1")
    axes[0].set(xlabel="Überlagerte Einträge N", ylabel="HRR-Auswahlgenauigkeit", ylim=(-.04, 1.04))
    axes[1].set(xlabel="Überlagerte Einträge N", ylabel="Störenergie nach Entbindung")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination / "superposition_capacity.png")
    plt.close(fig)
    packet = encode_text("Eine stehende Welle trägt exakt diese UTF-8-Bytes: Grüße 🌊.")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for current_time in (0., .01, .03, .07):
        snapshot = sample_wave(packet, current_time, points=1024)
        axes[0].plot(snapshot["displacement"], label=f"t={current_time}s")
        axes[1].plot(snapshot["quadrature"], label=f"t={current_time}s")
    axes[0].set_ylabel("Auslenkung q(x,t)")
    axes[1].set(xlabel="Räumlicher Abtastindex", ylabel="Quadratur p(x,t)")
    axes[0].legend(ncol=4)
    fig.tight_layout()
    fig.savefig(destination / "standing_wave.png")
    plt.close(fig)


def report(destination: Path, result: dict) -> None:
    codec, noise, invariants = result["codec"], result["noise"], result["invariants"]
    lines = ["# FreqAI: reproduzierbare numerische Ergebnisse", "",
        "Das Ergebnis ist ein trainingsfreier, exakt dekodierbarer Wellenspeicher mit lexikalischem Abruf. "
        "Die Berechnungen belegen keine natürliche physikalische Bedeutung von Text und keine allgemeine Sprachintelligenz.", "",
        "## Versuchsaufbau", "",
        f"Zeitpunkt UTC: {result['metadata']['generated_at_utc']}. Python {result['metadata']['python']}, "
        f"NumPy {result['metadata']['numpy']}, SciPy {result['metadata']['scipy']}.", "",
        "40 Speichertexte, 32 bekannte Fragen, 10 Nullfälle und 8 semantische Paraphrasen ohne gemeinsame Wörter. "
        "Fragetexte stehen ausschließlich in der Evaluierung, niemals im Speicher. Alle zwölf Kombinationen aus "
        "raw_bytes/words/hybrid und D=256/1024/4096/16384 werden vollständig ausgewertet. "
        "Schwelle 0,18 und hybride Gewichtung 75/25 sind feste Regeln; es gibt keine Optimierung von Modellparametern. "
        "Ein Leerprompt wird als abgewiesene Eingabe erfasst. Die anderen Nullfälle prüfen reguläre Nichtantworten. "
        "Der feste Wortüberlappungsfilter gehört zum System und wird in der Auswertung nicht umgangen.", "",
        "Die Konfigurationen werden auf demselben kleinen, manuell erstellten Testkorpus verglichen. "
        "Eine anhand dieser Ergebnisse gewählte Konfiguration besitzt damit noch keine unabhängig bestätigte Generalisierungsleistung. "
        "Es werden weder Synonymtabellen noch vortrainierte Embeddings noch Sprachmodelle verwendet.", "",
        "## Umkehrbare Textkodierung", "",
        "UTF-8-Bytes werden affinlinear auf [-1,1] abgebildet und orthonormal mit DCT-II transformiert. "
        "Jeder Koeffizient steuert einen stehenden Kosinusmodus. Die normierten Quadraturen "
        "q_k(t)=a_k cos(ω_k t), p_k(t)=a_k sin(ω_k t) erlauben a_k=q_k cos(ω_k t)+p_k sin(ω_k t). "
        "Inverse DCT, Rundung und UTF-8-Dekodierung rekonstruieren den Text; SHA-256 erkennt beschädigte Nutzdaten. "
        "Die Frequenzskalierung ist frei gewählt, kein entdecktes Naturgesetz.", "",
        f"Exakte Textrekonstruktion: **{sum(row['exact_text'] for row in codec)}/{len(codec)}** Versuche, "
        f"einschließlich Leertext, Nullbytes, Unicode, Zufallstexten und t bis 10^12 s. "
        f"Rekonstruktion aus vollständigen räumlichen Quadraturgittern: **{sum(row['full_grid_exact_text'] for row in codec)}/{len(codec)}**.", "",
        f"Maximaler Koeffizientenfehler: {max(row['coefficient_max_error'] for row in codec):.3e}. "
        f"Maximaler relativer Fehler der normierten Quadraturenergie: {max(row['relative_energy_error'] for row in codec):.3e}; "
        f"der frequenzgewichteten physikalischen Modenenergie: {max(row['relative_physical_energy_error'] for row in codec):.3e}. "
        f"Parseval-Fehler: {max(row['parseval_relative_error'] for row in codec):.3e}.", "",
        "Diese Langzeittests verwenden bekannte identische Zeitstempel für Vorwärts- und Rücktransformation. "
        "Sie belegen keine Uhrengenauigkeit eines realen Oszillators bei 10^12 s. Es wird analytisch ausgewertet, "
        "nicht über 10^12 Sekunden numerisch integriert. Nur die Auslenkung reicht an Nulldurchgängen nicht aus; "
        "drei gesonderte Versuche zeigen einen unsichtbaren Modus bei vollständig erhaltener Quadratur. "
        "Die Anzeige darf heruntergesampelt werden, zur verlustfreien Dekodierung bleiben alle Modi gespeichert. "
        "Float64-Koeffizienten benötigen roh acht Bytes je UTF-8-Byte zuzüglich Metadaten; die Darstellung ist keine Kompression.", "",
        "## Interferenz und Abruf", "",
        "Für normierte Fourier-Schlüssel K_i und Prompt Q liefert der kohärente Kreuzterm "
        "S_i=(||K_i+Q||²-||K_i||²-||Q||²)/2=Re⟨K_i,Q⟩ den Vergleichswert. "
        "Nach Parseval entspricht er exakt dem Skalarprodukt der ursprünglichen Merkmalsvektoren. "
        "Eine gemeinsame Phasenentwicklung verändert dieses Ergebnis nicht. Die Fourierdarstellung "
        "erzeugt deshalb aus sich heraus keine neue Semantik. Der ausgewählte Antworttext wird aus seinen stehenden Modi dekodiert.", "",
        f"Maximaler Unterschied zu direkter Kosinusähnlichkeit: {max(row['direct_cosine_max_error'] for row in invariants):.3e}. "
        f"Maximaler Unterschied zur expliziten Interferenzenergieberechnung: {max(row['explicit_interference_max_error'] for row in invariants):.3e}. "
        f"Zeitlich unveränderte Antworten: **{sum(row['answer_invariant'] for row in invariants)}/{len(invariants)}**.", "",
        "Der erste vollständige Lauf zeigte zwei zeitabhängige Antwortwechsel bei mathematisch gleichen Scores "
        "(words, D=256, Japan-Frage). Die numerische Abweichung lag bei 5,55e-17. "
        "Sortierung und Schwellwertentscheidung verwenden deshalb jetzt einheitlich auf zwölf Dezimalstellen "
        "gerundete Lesewerte (Auflösung 1e-12); die ausgegebenen Rohscores bleiben unverändert. "
        "Stabile Sortierung erhält bei Gleichstand die ursprüngliche Dokumentreihenfolge. "
        "Die negative Messung wurde unter ../iterations/01_before_tie_fix.json und .md aufbewahrt. "
        "Diese Korrektur stabilisiert die Numerik, sie verbessert nicht das Sprachverständnis.", "",
        "| Merkmale | D | Bekannte Fragen | Nullfälle korrekt | Paraphrasen korrekt |",
        "|---|---:|---:|---:|---:|"]
    for mode in ("raw_bytes", "words", "hybrid"):
        for dimensions in (256, 1024, 4096, 16384):
            grouped = {row["group"]: row for row in result["retrieval_summary"]
                       if row["feature_mode"] == mode and row["dimensions"] == dimensions}
            cells = [f"{grouped[group]['correct']}/{grouped[group]['cases']}" for group in ("known", "null", "semantic_no_overlap")]
            lines.append(f"| {mode} | {dimensions} | {' | '.join(cells)} |")
    lines.extend(["", "Die Paraphrasenprüfung ist absichtlich streng: semantisch verwandte Formulierungen besitzen "
        "keine identischen Wörter. Ein fester Wortüberlappungsfilter kann hier keine richtige Antwort auswählen. "
        "Die Roh-Rangfolge ohne Antwortfilter ist zusätzlich in retrieval_summary.csv und den Einzelfällen ausgewiesen. "
        "Ein Nullfall mit bekannten Stichwörtern kann trotzdem eine falsche gespeicherte Stelle aktivieren. "
        "Die Schwelle ist kein zuverlässiger Wissens- oder Wahrheitsdetektor.", "",
        "### Fehler der Referenzkonfiguration hybrid, D=4096", ""])
    failures = [row for row in result["retrieval_cases"] if row["feature_mode"] == "hybrid"
                and row["dimensions"] == 4096 and not row["correct"]]
    for row in failures:
        lines.append(f"- {row['case_id']} ({row['group']}): {row['prompt']!r}; erwartet {row['expected_id']}, "
                     f"gewählt {row['selected_id']}, höchster Score {row['top_score']:.4f}.")
    lines.extend(["", "## Rauschen und endliche Speicherkapazität", "",
        f"{len(noise)} Versuche addieren Gaußrauschen zu den Textkoeffizienten; je Rauschstärke 32 feste Seeds. "
        f"Nicht erkannte falsche Texte: **{sum(row['undetected_wrong_text'] for row in noise)}**. "
        "Eine Prüfsumme korrigiert Fehler nicht: oberhalb der Rundungsreserve wird die Dekodierung abgewiesen. "
        "noise.csv enthält jeden Versuch, die Grafik vergleicht die Messung mit der unabhängigen Gauß-Rundungswahrscheinlichkeit.", "",
        "Eine zusätzliche HRR-Simulation überlagert 8, 32, 128 oder 512 zufällige Einträge in nur einem gemeinsamen "
        "Spektrum bei D=256/1024/4096 und drei Seeds. Nach Entbindung wächst die gemessene Störenergie ungefähr "
        "mit N-1. Die Auswahl benötigt weiterhin ein separat gespeichertes Wörterbuch der Kandidaten. "
        "Dies ist keine verlustfreie Alternative zum adressierten Hauptspeicher. "
        "hrr_capacity.csv zeigt auch die verlorenen Einträge; hash_collisions.csv misst Kollisionen von 2000 künstlichen Tokens.", "",
        "Der Hauptspeicher vermeidet die verlustreiche Entmischung, indem er Dokumentadressen und alle Nutzdatenmodi behält. "
        "Seine Größe steigt mit der Datenmenge. Eine einzige endliche Welle ohne Adressen, Wörterbuch oder "
        "zusätzliche Struktur kann nicht beliebig viele unabhängige Texte verlustfrei speichern.", "",
        "## Ergebnis und Reproduktion", "",
        "Gefunden wurde eine numerisch überprüfte Lösung für Text → stehende Welle → Text und einen kohärenten, "
        "trainingsfreien Abruf aus einem endlichen Korpus. Nicht gelöst wurde die weitergehende Forderung nach "
        "allgemeinem Textverständnis oder frei erzeugten richtigen Antworten allein aus physikalischer Schwingung. "
        "Diese Einschränkung darf nicht durch längeres Wiederholen gleicher Rechnungen als gelöst dargestellt werden.", "",
        "Aufruf aus dem Projektverzeichnis: `.venv\\Scripts\\python.exe experiments\\run_experiments.py`. "
        "Die Dateien werden reproduzierbar neu erzeugt; Zeitstempel und Laufzeiten können abweichen. "
        "Der Benchmark-Hash und die Versionsangaben stehen in results.json. "
        "Numerische Integritätsverletzungen führen zu Exitcode 1; schwache Abrufleistung wird vollständig berichtet.", "",
        "Dateien: codec.csv, noise.csv, position_only.csv, retrieval_cases.csv, retrieval_summary.csv, "
        "invariants.csv, hrr_capacity.csv, hash_collisions.csv, results.json sowie vier PNG-Abbildungen."])
    (destination / "report_de.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "numerical")
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Explicit external evaluation corpus; no imported corpus copy is kept in the project")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark_path = args.benchmark
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    validate_benchmark(benchmark)
    started = time.perf_counter()
    codec, noise, position_only = codec_experiments()
    print(f"Codec: {sum(row['exact_text'] for row in codec)}/{len(codec)} exact; {len(noise)} noise trials", flush=True)
    cases, summaries, invariants = retrieval_experiments(benchmark)
    capacity = hrr_capacity()
    collisions = hash_collisions()
    result = {
        "metadata": {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                     "python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform(),
                     "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
                     "source_sha256": {name: hashlib.sha256((ROOT / "freqai" / name).read_bytes()).hexdigest()
                                       for name in ("codec.py", "features.py", "memory.py")},
                     "fixed_noise_seeds": list(range(32)), "fixed_hrr_seeds": [0, 1, 2],
                     "elapsed_seconds": time.perf_counter() - started,
                     "no_training": True, "independent_holdout_claimed": False},
        "codec": codec, "noise": noise, "position_only": position_only,
        "retrieval_cases": cases, "retrieval_summary": summaries, "invariants": invariants,
        "hrr_capacity": capacity, "hash_collisions": collisions,
    }
    checks = {
        "all_text_roundtrips_exact": all(row["exact_text"] and row["full_grid_exact_text"] for row in codec),
        "energy_conserved": max(row["relative_energy_error"] for row in codec) < 1e-12,
        "physical_energy_conserved": max(row["relative_physical_energy_error"] for row in codec) < 1e-12,
        "parseval_holds": max(row["parseval_relative_error"] for row in codec) < 1e-12,
        "time_invariant_answers": all(row["answer_invariant"] for row in invariants),
        "retrieval_equals_direct_cosine": max(row["direct_cosine_max_error"] for row in invariants) < 1e-12,
        "interference_identity_holds": max(row["explicit_interference_max_error"] for row in invariants) < 1e-12,
        "no_undetected_noise_corruption": not any(row["undetected_wrong_text"] for row in noise),
    }
    result["numerical_checks"] = checks
    for name, rows in result.items():
        if isinstance(rows, list):
            write_csv(args.output / f"{name}.csv", rows)
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plots(args.output, summaries, noise, capacity)
    report(args.output, result)
    print(json.dumps({"numerical_checks": checks, "output": str(args.output)}, indent=2), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
