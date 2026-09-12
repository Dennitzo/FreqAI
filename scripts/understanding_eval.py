"""Language understanding measured on the central dataset, per expansion step.

Six axes, all automatic and auditable, because a human user study is not part of
this project. Every item keeps its prompt, answer and the exact reason for its
score, so no number here is a black box.

``addressability``      Can a declaration that exists in the corpus be addressed
                        by a question at all? Probes are built from sentences that
                        entered with this step only.
``definitions``         The 40 known information questions with their rubric
                        concept groups (measured, never tuned on).
``binding``             Forward and reverse attribute binding of synthetic random
                        facts (mechanism control on its own small corpus).
``dialogue_echo``       Do conversation answers add content or only echo the user?
``unknown_abstention``  Does the system stay silent about names it was never told?
``determinism``         Same prompt, same answer.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import re
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/understanding"
SEED = 20260911


def load_corpus(path: Path):
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                records.append({"id": item["id"], "text": item["text"], "source": item.get("source", "")})
    return records


def content_tokens(text):
    from freqai.information import STOPWORDS, tokenize
    return {token for token in tokenize(text)
            if token not in STOPWORDS and (token.isalpha() or any(char.isdigit() for char in token))}


def overlap(answer_tokens, reference_tokens):
    if not reference_tokens:
        return 0.0
    return len(answer_tokens & reference_tokens) / len(reference_tokens)


DEFINITION_SENTENCE = re.compile(r"^(?:Der|Die|Das)\s[^.!?]{3,80}?\s(?:ist|sind)\s[^.!?]{6,240}[.]$", re.U)


def addressability_probes(new_records, limit=60):
    """Questions built from declarations that entered with this step only."""
    from freqai.information import _sentences, _subject
    candidates = []
    for record in new_records:
        for sentence in _sentences(record["text"])[:1]:
            if not DEFINITION_SENTENCE.match(sentence):
                continue
            concept = _subject(sentence)
            if not concept or len(concept.split()) > 6:
                continue
            candidates.append({"concept": concept, "sentence": sentence})
    if not candidates:
        return []
    rng = np.random.default_rng(SEED)
    picks = [candidates[i] for i in sorted(rng.choice(len(candidates), size=min(limit, len(candidates)),
                                                     replace=False))]
    return [{"id": f"address-{index:03d}", "prompt": f"Was ist {pick['concept']}?",
             "reference": pick["sentence"]} for index, pick in enumerate(picks)]


def run_axis(model, items, reference_key=None):
    rows = []
    for item in items:
        started = time.perf_counter()
        output = model.generate(item["prompt"])
        answer_tokens = content_tokens(output["text"])
        row = {"id": item["id"], "prompt": item["prompt"], "answer": output["text"],
               "answered": bool(output["tokens"]),
               "seconds": round(time.perf_counter() - started, 4)}
        if reference_key and item.get(reference_key):
            row["coverage"] = round(overlap(answer_tokens, content_tokens(item[reference_key])), 3)
            row["addressed"] = row["coverage"] >= 0.5
        rows.append(row)
    return rows


def axis_definitions(model, path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for case in payload["cases"]:
        output = model.generate(case["prompt"])
        groups = [content_tokens(" ".join(group)) for group in case.get("concepts", [])]
        answer_tokens = content_tokens(output["text"])
        hits = sum(1 for group in groups if overlap(answer_tokens, group) >= 0.5)
        rows.append({"id": case["id"], "prompt": case["prompt"], "answer": output["text"],
                     "answered": bool(output["tokens"]), "concept_groups": len(groups),
                     "groups_hit": hits})
    answered = [row for row in rows if row["answered"]]
    return {"items": len(rows), "answered": len(answered),
            "full_concept_match": sum(1 for row in answered if row["groups_hit"] == row["concept_groups"]),
            "rows": rows}


def randomfact_records():
    """The synthetic fact corpus lives in the evaluation split and never in the dataset."""
    payload = json.loads((ROOT / "memory/evaluation/information_randomfacts.json").read_text(encoding="utf-8"))
    return [{"id": item["id"], "text": item["text"], "source": item["source"]} for item in payload["records"]]


def axis_binding(model_path_records):
    from freqai.unpaired import UnpairedWaveModel
    model = UnpairedWaveModel(model_path_records, order=2)
    payload = json.loads((ROOT / "memory/evaluation/information_randomfacts.json")
                         .read_text(encoding="utf-8"))
    rows = []
    for case in payload["cases"]:
        output = model.generate(case["prompt"])
        text = output["text"].casefold()
        if case["direction"] in {"entity_to_value", "value_to_entity"}:
            ok = str(case["expected_value"]) in text and case["expected_entity"].casefold() in text
        elif case["direction"] == "unknown":
            ok = not output["text"].strip()
        else:
            ok = None
        rows.append({"id": case["id"], "direction": case["direction"], "prompt": case["prompt"],
                     "answer": output["text"], "ok": ok})
    scored = [row for row in rows if row["ok"] is not None]
    return {"items": len(rows), "scored": len(scored), "passed": sum(bool(row["ok"]) for row in scored),
            "rows": rows}


def axis_dialogue(model, path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scenario in payload["scenarios"]:
        context = None
        for turn in scenario["turns"]:
            output = model.generate(turn["prompt"], context=context)
            context = output.get("state")
            user_tokens = content_tokens(turn["prompt"])
            answer_tokens = content_tokens(output["text"])
            rows.append({"id": turn["id"], "scenario": scenario["id"], "prompt": turn["prompt"],
                         "answer": output["text"], "answered": bool(output["tokens"]),
                         "echo_share": round(overlap(answer_tokens, user_tokens), 3) if answer_tokens else 0.0})
    answered = [row for row in rows if row["answered"]]
    return {"items": len(rows), "answered": len(answered),
            "echo_answers": sum(1 for row in answered if row["echo_share"] >= 0.6),
            "rows": rows}


def axis_unknown(model, probes):
    rows = []
    for item in probes:
        output = model.generate(item["prompt"])
        rows.append({"id": item["id"], "prompt": item["prompt"], "answer": output["text"],
                     "abstained": not bool(output["tokens"])})
    return {"items": len(rows), "abstained": sum(row["abstained"] for row in rows), "rows": rows}


def axis_determinism(model, prompts):
    rows = []
    for prompt in prompts:
        first = model.generate(prompt)["text"]
        second = model.generate(prompt)["text"]
        rows.append({"prompt": prompt, "stable": first == second})
    return {"items": len(rows), "stable": sum(row["stable"] for row in rows), "rows": rows}


def unknown_name_probes(model, count=24):
    """Names that exist in no declaration of this corpus."""
    rng = np.random.default_rng(SEED)
    letters = "aeiou" + "kbdmnrstvglzpfh"
    probes, tried = [], set()
    while len(probes) < count:
        length = int(rng.integers(5, 8))
        name = "".join(rng.choice(list("kbdmnrstvglzp")) if i % 2 == 0 else rng.choice(list("aeiou"))
                       for i in range(length)).capitalize()
        if name in tried:
            continue
        tried.add(name)
        if name.lower() in model.index:
            continue
        probes.append({"id": f"unknown-{len(probes):02d}", "prompt": f"Welche Frequenz hat das Prüfsignal {name}?"})
    return probes


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-file", type=Path, required=True)
    parser.add_argument("--previous-step-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from freqai.unpaired import UnpairedWaveModel

    records = load_corpus(args.step_file)
    started = time.perf_counter()
    model = UnpairedWaveModel(records, order=2)
    build_seconds = round(time.perf_counter() - started, 1)

    new_records = records
    if args.previous_step_file:
        known = {record["id"] for record in load_corpus(args.previous_step_file)}
        new_records = [record for record in records if record["id"] not in known]

    probes = addressability_probes(new_records)
    result = {
        "corpus": {"file": args.step_file.name, "records": len(records),
                   "new_records_in_step": len(new_records), "build_seconds": build_seconds,
                   "vocabulary_size": model.size, "prefix_fields": len(model.transition_spectra),
                   "data_wave_modes": model.data_wave().mode_count},
        "axes": {
            "addressability": {"items": len(probes), **(lambda rows: {
                "answered": sum(row["answered"] for row in rows),
                "addressed": sum(row.get("addressed", False) for row in rows),
                "mean_coverage": round(sum(row.get("coverage", 0.0) for row in rows)/max(1, len(rows)), 3),
                "rows": rows})(run_axis(model, probes, reference_key="reference"))},
            "definitions": axis_definitions(model, ROOT / "memory/evaluation/unpaired_knowledge_regression.json"),
            "binding": axis_binding(randomfact_records()),
            "dialogue_echo": axis_dialogue(model, ROOT / "memory/evaluation/unpaired_chat_development.json"),
            "unknown_abstention": axis_unknown(model, unknown_name_probes(model)),
            "determinism": axis_determinism(model, ["Was ist eine Frequenz?", "Wie heißt du?",
                                                    "Mit welcher Einheit gibt man die Frequenz an?"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {name: {key: value for key, value in axis.items() if key != "rows"}
               for name, axis in result["axes"].items()}
    print(json.dumps({"corpus": result["corpus"], "axes": summary}, ensure_ascii=False, indent=1))



if __name__ == "__main__":
    main(sys.argv[1:])
