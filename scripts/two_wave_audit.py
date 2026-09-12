"""Numerical audit of the two-wave readout on the real corpus (never imported as data).

Every decoding state reachable with a set of evaluation prompts, over all interference
controls: agreement of the spectral operator and the direct readout, normalization and
energy invariants, and how often each of the two waves is causally visible.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/two_wave_interference"

#: (prompt_gain, data_gain, phase_error) settings that are all walked.
CONTROLS = [(1.0, 0.15, 0.0), (0.0, 0.15, 0.0), (1.0, 0.0, 0.0), (1.0, 0.5, 1.7),
            (1.0, 0.15, np.pi)]


def evaluation_prompts():
    prompts = [case["prompt"] for case in
               json.loads((ROOT / "memory/evaluation/unpaired_knowledge_regression.json").read_text(encoding="utf-8"))["cases"]]
    chat = json.loads((ROOT / "memory/evaluation/unpaired_chat_development.json").read_text(encoding="utf-8"))
    prompts += [turn["prompt"] for scenario in chat["scenarios"][:6] for turn in scenario["turns"]]
    return prompts


def main():
    from freqai.store import MemoryStore
    from freqai.unpaired import UnpairedWaveModel
    from freqai.waves import VERIFY_TOLERANCE, resonance

    store = MemoryStore(ROOT / "memory/memory.sqlite3")
    _, documents = store.snapshot()
    records = [{"id": d.id, "text": d.text, "source": d.source} for d in documents]
    started = time.time()
    model = UnpairedWaveModel(records, order=2)
    build_seconds = time.time() - started

    data_wave = model.data_wave()
    states = 0
    max_error = 0.0
    worst = None
    multi_mode = single_mode = coupled_prompt = coupled_data = 0
    resonance_values = []
    failures = []

    for prompt in evaluation_prompts():
        field, _ = model.prompt_field(prompt)
        try:
            answer = model.generate(prompt)
        except Exception as error:
            failures.append({"prompt": prompt, "error": str(error)})
            continue
        resonance_values.append(resonance(data_wave, model.prompt_wave(field)))
        for length in range(min(len(answer["tokens"]), 6) + 1):
            prefix = answer["tokens"][:length]
            for prompt_gain, data_gain, phase_error in CONTROLS:
                for policy in ("deepest", "all"):
                    call = dict(method="operator", prompt_gain=prompt_gain, data_gain=data_gain,
                                phase_error=phase_error, order_policy=policy)
                    try:
                        operator_p, trace = model.next_distribution(prefix, field, **call)
                        direct_p, _ = model.next_distribution(prefix, field, **{**call, "method": "direct"})
                    except ValueError:
                        continue
                    states += 1
                    difference = float(trace["fft_roundtrip_max_error"])
                    max_error = max(max_error, difference)
                    if worst is None or difference > worst["max_error"]:
                        worst = {"max_error": difference, "prompt": prompt, "prefix": list(prefix),
                                 "order_policy": policy, "prompt_gain": prompt_gain,
                                 "data_gain": data_gain, "phase_error": float(phase_error)}
                    for name, distribution in (("operator", operator_p), ("direct", direct_p)):
                        if not np.isfinite(distribution).all() or abs(float(distribution.sum()) - 1.0) > 1e-9:
                            failures.append({"prompt": prompt, "prefix": list(prefix), "method": name})
                    if trace["interference_verification"]["max_error"] > VERIFY_TOLERANCE:
                        failures.append({"prompt": prompt, "problem": "tolerance"})
                    active = np.array(trace["active_token_indices"], dtype=np.intp)
                    if len(active) <= 1:
                        single_mode += 1
                        continue
                    multi_mode += 1
                    off_prompt, _ = model.next_distribution(prefix, field, **{**call, "prompt_gain": 0.0})
                    off_data, _ = model.next_distribution(prefix, field, **{**call, "data_gain": 0.0})
                    coupled_prompt += bool(0.5 * float(np.abs(operator_p - off_prompt).sum()) > 1e-12)
                    coupled_data += bool(0.5 * float(np.abs(operator_p - off_data).sum()) > 1e-12)

    result = {
        "corpus": {"documents": len(records), "vocabulary_size": model.size,
                   "build_seconds": round(build_seconds, 2)},
        "data_wave": {"mode_count": data_wave.mode_count, "carrier_size": data_wave.carrier_size,
                      "energy": data_wave.energy, "fingerprint": data_wave.fingerprint()},
        "audited_states": states,
        "max_operator_direct_error": max_error,
        "tolerance": VERIFY_TOLERANCE,
        "worst_state": worst,
        "single_mode_states": single_mode,
        "multi_mode_states": multi_mode,
        "coupled_by_prompt_wave": coupled_prompt,
        "coupled_by_data_wave": coupled_data,
        "prompt_data_resonance_mean": float(np.mean(resonance_values)) if resonance_values else 0.0,
        "invariant_failures": failures,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numerical_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "worst_state"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
