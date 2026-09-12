"""Sealed synthetic factual direction/coupling test, declarations only.

Random entities and values are authored once before candidate outputs. Questions
never reach the model constructor; values are permuted in a separate rebuild to
test whether generated content follows data rather than memorized answer code.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random
import re
import sys
import time

import evaluate_information_corpus as evaluation
import evaluate_public_corpus as common

OUTPUT = evaluation.OUTPUT
SPEC = evaluation.ROOT / "memory/evaluation/information_randomfacts.json"


def initialize():
    if SPEC.exists():
        return common.read(SPEC)
    rng = random.Random(830117)
    names = []
    while len(names) < 12:
        name = "".join(rng.choice("bcdfgklmnprstvz") + rng.choice("aeiou") for _ in range(3)).capitalize()
        if name not in names:
            names.append(name)
    values = rng.sample(range(113, 977), len(names))
    facts = []
    cases = []
    for i, (name, value) in enumerate(zip(names, values)):
        facts.append({"id": f"random-fact-{i:02}", "title": name, "prompt": "",
                      "source": "evaluation synthetic random declarative facts, never production import",
                      "text": f"{name} ist ein periodisches Prüfsignal. Die Frequenz von {name} beträgt {value} Hertz.",
                      "evaluation_truth": {"entity": name, "frequency_hz": value}})
        cases.append({"id": f"rf{i:02}", "direction": "entity_to_value", "entity_index": i,
                      "prompt": f"Welche Frequenz hat das Prüfsignal {name}?", "expected_entity": name, "expected_value": value})
        cases.append({"id": f"rr{i:02}", "direction": "value_to_entity", "entity_index": i,
                      "prompt": f"Welches Prüfsignal hat eine Frequenz von {value} Hertz?", "expected_entity": name, "expected_value": value})
    for i in range(4):
        cases.append({"id": f"rn{i:02}", "direction": "negation", "entity_index": i,
                      "prompt": f"Hat {names[i]} eine Frequenz von {values[i+1]} Hertz?",
                      "expected_entity": names[i], "expected_value": values[i], "rejected_value": values[i+1]})
    for i, unknown in enumerate(("Javuro", "Tepali", "Wemuko", "Xarifo")):
        assert unknown not in names
        cases.append({"id": f"ru{i:02}", "direction": "unknown", "prompt": f"Welche Frequenz hat das Prüfsignal {unknown}?",
                      "expected_entity": unknown, "expected_value": None})
    value = {"version": 1, "seed": 830117, "created_utc": common.timestamp(), "records": facts, "cases": cases,
             "protocol": "All input records are declarative prose with empty prompt. evaluation_truth and cases never reach constructor. 12 forward,12 reverse,4 negation,4 unknown. No training or tuned mix/source gains.",
             "permutation": "Rotate all frequency values one position among the same entity titles, then repeat12 entity-to-value questions. Success requires the output to follow changed facts.",
             "seal": "Do not generate answers until final candidate code freeze. This is a small synthetic capability diagnosis, not natural language generalization proof."}
    evaluation.immutable(SPEC, value)
    evaluation.immutable(OUTPUT / "randomfacts_seal.json", {"created_utc": common.timestamp(), "spec_sha256": common.sha(SPEC)})
    return value


def clean_records(spec, permute=False):
    records = []
    values = [r["evaluation_truth"]["frequency_hz"] for r in spec["records"]]
    for i, row in enumerate(spec["records"]):
        clean = {key: row[key] for key in ("id", "title", "prompt", "source", "text")}
        if permute:
            name = row["evaluation_truth"]["entity"]
            clean["text"] = f"{name} ist ein periodisches Prüfsignal. Die Frequenz von {name} beträgt {values[(i+1)%len(values)]} Hertz."
        records.append(clean)
    return records


def proxy(case, answer, all_names):
    mentioned = {n for n in all_names if n.casefold() in answer.casefold()}
    numbers = {int(n) for n in re.findall(r"(?<!\w)\d+(?!\w)", answer)}
    direction = case["direction"]
    if direction == "entity_to_value":
        passed = numbers == {case["expected_value"]} and not (mentioned - {case["expected_entity"]})
    elif direction == "value_to_entity":
        passed = mentioned == {case["expected_entity"]} and not (numbers - {case["expected_value"]})
    elif direction == "negation":
        passed = bool(re.search(r"\b(nein|nicht|keine)\b", answer.casefold())) and case["expected_value"] in numbers
    else:
        passed = not numbers and (not answer.strip() or bool(re.search(r"\b(unbekannt|keine|nicht|fehlt|unbekannten)\b", answer.casefold())))
    return {"strict_lexical_proxy": passed, "mentioned_entities": sorted(mentioned), "numbers": sorted(numbers),
            "semantic_review_required": True}


def signature(model):
    method = getattr(model, "model_signature", None)
    if callable(method):
        return method()
    return {"available": False, "reason": "No deterministic coefficient signature API provided"}


def run(args):
    spec = initialize()
    assert common.sha(SPEC) == common.read(OUTPUT / "randomfacts_seal.json")["spec_sha256"]
    freeze = common.read(OUTPUT / f"{args.freeze_label}_freeze.json")
    code_root = Path(args.code_root).resolve()
    assert common.code_manifest(code_root) == freeze["manifest"]["code"], "Code changed after candidate freeze"
    destination = OUTPUT / f"{args.label}_randomfacts.json"
    assert not destination.exists()
    sys.path.insert(0, str(code_root))
    InformationWaveModel = importlib.import_module("freqai.information").InformationWaveModel
    records = clean_records(spec)
    model = InformationWaveModel(records, order=2)
    rebuilt = InformationWaveModel(records, order=2)
    original_signature, rebuild_signature = signature(model), signature(rebuilt)
    all_names = [row["evaluation_truth"]["entity"] for row in spec["records"]]
    rows = []
    for case in spec["cases"]:
        start = time.perf_counter()
        actual = model.generate(case["prompt"], max_tokens=64, decoding="beam", seed=17, method="operator")
        repeated = rebuilt.generate(case["prompt"], max_tokens=64, decoding="beam", seed=17, method="operator")
        rows.append({**case, "answer": actual["text"], "tokens": actual.get("tokens", []),
                     "elapsed_two_builds_s": time.perf_counter()-start,
                     "deterministic_rebuild_output": actual["text"] == repeated["text"] and actual.get("tokens") == repeated.get("tokens"),
                     "proxy": proxy(case, actual["text"], all_names), "qualitative_review": None})
    permuted_model = InformationWaveModel(clean_records(spec, permute=True), order=2)
    values = [r["evaluation_truth"]["frequency_hz"] for r in spec["records"]]
    permuted_rows = []
    for case in (row for row in spec["cases"] if row["direction"] == "entity_to_value"):
        changed = {**case, "expected_value": values[(case["entity_index"]+1)%len(values)]}
        actual = permuted_model.generate(case["prompt"], max_tokens=64, decoding="beam", seed=17, method="operator")
        permuted_rows.append({**changed, "answer": actual["text"], "tokens": actual.get("tokens", []),
                              "proxy": proxy(changed, actual["text"], all_names), "qualitative_review": None})
    result = {"created_utc": common.timestamp(), "code": common.code_manifest(code_root), "spec_sha256": common.sha(SPEC),
              "freeze_sha256": common.sha(OUTPUT / f"{args.freeze_label}_freeze.json"),
              "declarative_documents": len(records), "paired_documents": 0, "optimization_steps": 0,
              "model_signature": original_signature, "rebuild_signature": rebuild_signature,
              "signatures_equal": original_signature == rebuild_signature,
              "rows": rows, "permuted_rows": permuted_rows,
              "summary": {"deterministic_rebuild_outputs": sum(row["deterministic_rebuild_output"] for row in rows),
                          "strict_lexical_proxy_by_direction": {d: sum(r["proxy"]["strict_lexical_proxy"] for r in rows if r["direction"] == d)
                                                               for d in ("entity_to_value", "value_to_entity", "negation", "unknown")},
                          "permuted_data_following_proxy": sum(row["proxy"]["strict_lexical_proxy"] for row in permuted_rows),
                          "warning": "Proxy success is provisional until semantic review; generated source sentences do not demonstrate free composition."}}
    evaluation.immutable(destination, result)
    print(json.dumps({"path": str(destination), "summary": result["summary"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["initialize", "run"])
    parser.add_argument("--code-root", default=str(evaluation.ROOT))
    parser.add_argument("--freeze-label", default="final")
    parser.add_argument("--label", default="final")
    args = parser.parse_args()
    if args.action == "initialize":
        spec = initialize()
        print(json.dumps({"spec": str(SPEC), "sha256": common.sha(SPEC), "records": len(spec["records"]), "cases": len(spec["cases"])}))
    else:
        run(args)


if __name__ == "__main__":
    main()
