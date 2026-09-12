"""Independent causal ablations of both prompt channels and stored data waves.

These experiments establish numerical dependence, not linguistic quality.
Only the development split is used. No stored complete-answer lookup is used
to calculate any numerical expected result.
"""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import inspect
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_generative_waves as evaluation


def difference(base, alternative):
    assert np.isfinite(alternative).all()
    assert np.min(alternative) >= -1e-14
    assert abs(float(np.sum(alternative)) - 1) < 1e-12
    distance = float(0.5 * np.abs(base - alternative).sum())
    return {"total_variation": distance, "changed_distribution": distance > 1e-10,
            "changed_argmax": int(np.argmax(np.round(base, 12))) != int(np.argmax(np.round(alternative, 12))),
            "raw_argmax_changed": int(np.argmax(base)) != int(np.argmax(alternative)),
            "max_absolute_error": float(np.max(np.abs(base - alternative)))}


def run() -> tuple[Path, dict]:
    from freqai.generative import PromptField
    from freqai.generation_runtime import generator_for
    memory, _ = evaluation.engine()
    model = generator_for(memory).model
    rng = np.random.default_rng(624813)
    data_phase = np.exp(2j * np.pi * rng.random(model.size))
    scrambled = copy.copy(model)
    scrambled.transition_spectra = {key: value * data_phase for key, value in model.transition_spectra.items()}
    scrambled.conditioned_spectra = {key: value * data_phase for key, value in model.conditioned_spectra.items()}
    null_memory = copy.copy(model)
    null_memory.transition_spectra = {key: np.zeros_like(value) for key, value in model.transition_spectra.items()}
    null_memory.conditioned_spectra = {key: np.zeros_like(value) for key, value in model.conditioned_spectra.items()}
    maximum_amplitude_error = max(float(np.max(np.abs(np.abs(value) - np.abs(scrambled.transition_spectra[key]))))
                                  for key, value in model.transition_spectra.items())
    rows = []
    direct_available = '"direct"' in inspect.getsource(model.next_distribution) or "'direct'" in inspect.getsource(model.next_distribution)
    for scenario in evaluation.suite():
        if scenario["split"] != "development":
            continue
        for turn in scenario["turns"]:
            field, _ = model.prompt_field(turn["prompt"])
            output = model.generate(turn["prompt"], decoding="beam", max_tokens=40, seed=17)
            prefixes = [[], output["tokens"][:3], output["tokens"][:8]]
            for prefix in prefixes:
                base, trace = model.next_distribution(prefix, field)
                token_null = PromptField(np.zeros(model.size), field.feature_spectrum.copy())
                feature_null = PromptField(np.asarray(field).copy(), np.zeros_like(field.feature_spectrum))
                both_null = PromptField(np.zeros(model.size), np.zeros_like(field.feature_spectrum))
                feature_phase = np.exp(2j * np.pi * rng.random(len(field.feature_spectrum)))
                feature_scramble = PromptField(np.asarray(field).copy(), field.feature_spectrum * feature_phase)
                alternatives = {
                    "token_channel_null": model.next_distribution(prefix, token_null)[0],
                    "feature_channel_null": model.next_distribution(prefix, feature_null)[0],
                    "both_prompt_channels_null": model.next_distribution(prefix, both_null)[0],
                    "prompt_gain_zero": model.next_distribution(prefix, field, prompt_gain=0)[0],
                    "relative_phase_pi": model.next_distribution(prefix, field, phase_error=math.pi)[0],
                    "feature_spectral_phase_scramble": model.next_distribution(prefix, feature_scramble)[0],
                    "data_spectral_phase_scramble": scrambled.next_distribution(prefix, field)[0],
                    "time_1_375": model.next_distribution(prefix, field, time_s=1.375)[0],
                    "time_86400_125": model.next_distribution(prefix, field, time_s=86400.125)[0],
                }
                if direct_available:
                    alternatives["direct_calculation"] = model.next_distribution(prefix, field, method="direct")[0]
                try:
                    value, _ = null_memory.next_distribution(prefix, field)
                    null_result = {"explicit_refusal": False, "finite_values": bool(np.isfinite(value).all())}
                except ValueError as error:
                    null_result = {"explicit_refusal": True, "error": str(error)}
                rows.append({"id": turn["id"], "prefix": prefix,
                             "support_size": trace["support_size"],
                             "baseline_argmax": model.vocabulary[int(np.argmax(base))],
                             "ablations": {name: difference(base, value) for name, value in alternatives.items()},
                             "null_channels_equal_gain_zero": float(np.max(np.abs(
                                 alternatives["both_prompt_channels_null"] - alternatives["prompt_gain_zero"]))),
                             "null_memory": null_result})
    names = rows[0]["ablations"].keys()
    summary = {name: {"checked": len(rows),
                      "changed_distributions": sum(row["ablations"][name]["changed_distribution"] for row in rows),
                      "changed_argmax": sum(row["ablations"][name]["changed_argmax"] for row in rows),
                      "raw_argmax_changed": sum(row["ablations"][name]["raw_argmax_changed"] for row in rows),
                      "mean_total_variation": float(np.mean([row["ablations"][name]["total_variation"] for row in rows])),
                      "max_absolute_error": max(row["ablations"][name]["max_absolute_error"] for row in rows)}
               for name in names}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "manifest": evaluation.model_manifest(),
              "audit_sha256": evaluation.digest(Path(__file__)), "suite": "development only; 30 prompts, 3 prefixes each",
              "phase_scramble_seed": 624813, "direct_available": direct_available,
              "argmax_tie_policy": "Compare probabilities rounded to 12 decimal places; raw differences are reported separately.",
              "data_scramble_preserved_spectral_amplitudes_max_error": maximum_amplitude_error,
              "null_memory_explicit_refusals": sum(row["null_memory"]["explicit_refusal"] for row in rows),
              "all_zero_prompt_matches_zero_gain_max_error": max(row["null_channels_equal_gain_zero"] for row in rows),
              "summary": summary, "rows": rows,
              "interpretation": ["Distribution changes prove numerical influence, not semantic correctness.",
                                 "Separate token and semantic/lexical feature channels both count as prompt information.",
                                 "Null memory clears unconditional and conditioned complex spectra but leaves reference arrays and grammar supports intact.",
                                 "Unconditional output timing can remain invariant because shared mode phases disappear under intensity readout."]}
    path = evaluation.OUTPUT / ("numerics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    evaluation.write(path, report)
    return path, report


if __name__ == "__main__":
    import json
    path, report = run()
    print(json.dumps({"path": str(path), "summary": report["summary"],
                      "null_memory_explicit_refusals": report["null_memory_explicit_refusals"]}, indent=2))
