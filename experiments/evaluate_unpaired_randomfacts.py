"""Replay the already-known declarative random-fact diagnostic, without QA input.

This is a regression of the earlier information experiment, never a new holdout.
Questions and evaluation truth remain outside the model constructor.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import evaluate_public_corpus as common
import evaluate_information_randomfacts as historical

OUT = common.ROOT / "results/unpaired_system/evaluation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--label", default="final")
    args = parser.parse_args()
    destination = OUT / f"{args.label}_randomfacts_regression.json"
    assert not destination.exists()
    code_root = Path(args.code_root).resolve()
    freeze = common.read(OUT / f"{args.label}_freeze.json")
    assert common.code_manifest(code_root) == freeze["manifest"]["code"]
    spec = common.read(historical.SPEC)
    assert common.sha(historical.SPEC) == common.read(historical.OUTPUT / "randomfacts_seal.json")["spec_sha256"]
    sys.path.insert(0, str(code_root))
    Model = importlib.import_module("freqai.unpaired").UnpairedWaveModel
    records = historical.clean_records(spec)
    assert all(not row["prompt"] for row in records)
    model, rebuilt = Model(records, order=2), Model(records, order=2)
    signature, rebuilt_signature = model.model_signature(), rebuilt.model_signature()
    names = [row["evaluation_truth"]["entity"] for row in spec["records"]]
    rows = []
    for case in spec["cases"]:
        actual = model.generate(case["prompt"], max_tokens=64, decoding="beam", seed=17, method="operator")
        repeated = rebuilt.generate(case["prompt"], max_tokens=64, decoding="beam", seed=17, method="operator")
        rows.append({**case, "answer": actual["text"], "tokens": actual.get("tokens", []),
                     "diagnostics": {key: actual.get(key) for key in ("reason", "ended", "analysis", "tokenization")},
                     "deterministic_rebuild_output": actual["text"] == repeated["text"] and actual.get("tokens") == repeated.get("tokens"),
                     "proxy": historical.proxy(case, actual["text"], names), "qualitative_review": None})
    changed_model = Model(historical.clean_records(spec, permute=True), order=2)
    values = [row["evaluation_truth"]["frequency_hz"] for row in spec["records"]]
    permuted = []
    for case in (row for row in spec["cases"] if row["direction"] == "entity_to_value"):
        changed = {**case, "expected_value": values[(case["entity_index"]+1) % len(values)]}
        actual = changed_model.generate(case["prompt"], max_tokens=64, decoding="beam", seed=17, method="operator")
        permuted.append({**changed, "answer": actual["text"], "tokens": actual.get("tokens", []),
                         "proxy": historical.proxy(changed, actual["text"], names), "qualitative_review": None})
    result = {"created_utc": common.timestamp(), "status": "known regression, not a fresh holdout",
              "code": common.code_manifest(code_root), "spec_sha256": common.sha(historical.SPEC),
              "freeze_sha256": common.sha(OUT / f"{args.label}_freeze.json"), "runner_sha256": common.sha(__file__),
              "historical_helper_sha256": common.sha(historical.__file__), "records": len(records),
              "paired_records": 0, "optimization_steps": 0,
              "model_signature": signature, "rebuild_signature": rebuilt_signature,
              "permuted_model_signature": changed_model.model_signature(), "signatures_equal": signature == rebuilt_signature,
              "rows": rows, "permuted_rows": permuted,
              "summary": {"deterministic_rebuild_outputs": sum(row["deterministic_rebuild_output"] for row in rows),
                          "strict_lexical_proxy_by_direction": {direction: sum(row["proxy"]["strict_lexical_proxy"] for row in rows if row["direction"] == direction)
                              for direction in ("entity_to_value", "value_to_entity", "negation", "unknown")},
                          "permuted_data_following_proxy": sum(row["proxy"]["strict_lexical_proxy"] for row in permuted),
                          "warning": "Semantic review required; empty unknown refusals are not informative answers, and a copied factual clause is not free composition."}}
    common.write(destination, result)
    print(json.dumps({"path": str(destination), "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
