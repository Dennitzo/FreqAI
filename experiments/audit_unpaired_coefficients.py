"""Reproducible numerical controls, distinct from language-quality holdouts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from freqai.unpaired import UnpairedWaveModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"results/unpaired_system/numerical_audit.json")
    args = parser.parse_args()
    source = ROOT/"memory/information/conversation_facts.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    records.extend({"text":f"Das Signal {name} hat eine Frequenz von {value} Hertz."}
                   for name,value in [("Lumor",19),("Tavor",73),("Nivora",41)])
    model = UnpairedWaveModel(records)
    signature_before = model.model_signature()
    questions = ["Hallo", "Wie heißt du?", "Was kannst du?", "Hast du Gefühle?",
                 "Mir geht es gut, wie geht es dir?", "Ich bin müde.","Ich heiße Xyrländo.",
                 "Welche Frequenz hat das Signal Lumor?", "Welches Signal hat eine Frequenz von 73 Hertz?"]
    checked_states = 0
    maximum_error = 0.0
    rows = []
    for prompt in questions:
        field,_ = model.prompt_field(prompt)
        result = model.generate(prompt,max_tokens=64)
        assert result["tokens"]
        for position in range(len(result["tokens"])+1):
            prefix = result["tokens"][:position]
            direct,_ = model.next_distribution(prefix,field,method="direct")
            for time_s in (0.0,1.625,86400.125):
                operator,trace = model.next_distribution(prefix,field,method="operator",time_s=time_s)
                maximum_error = max(maximum_error,float(np.max(np.abs(direct-operator))))
                assert np.array_equal(direct,operator)
                assert trace["all_supported_modes_included"]
                checked_states += 1
        rows.append({"prompt":prompt,"answer":result["text"],"tokenization":result["tokenization"],
                     "whole_answer_occurs_in_input":any(result["text"] in row["text"] for row in records)})
    field,_ = model.prompt_field("Hallo")
    normal,_ = model.next_distribution([],field)
    inverted,_ = model.next_distribution([],field,phase_error=np.pi)
    phase_tv = float(np.abs(normal-inverted).sum()/2)
    assert phase_tv>.5 and np.argmax(normal)!=np.argmax(inverted)
    zero_prompt_refuses = False
    try:
        model.next_distribution([],field*0)
    except ValueError:
        zero_prompt_refuses = True
    assert zero_prompt_refuses
    phase_model = UnpairedWaveModel([{"text":"Der Assistent kann Texte lesen. Der Assistent kann Texte schreiben. Der Assistent kann Wörter zählen."}])
    role_field,_ = phase_model.prompt_field("Was kannst du?")
    role_before,_ = phase_model.next_distribution(["ich","kann"],role_field)
    phase_model.lexical_roles[phase_model.facts[0].object_role].spectrum *= -1
    role_after,_ = phase_model.next_distribution(["ich","kann"],role_field)
    role_phase_tv = float(np.abs(role_before-role_after).sum()/2)
    assert np.argmax(role_before)!=np.argmax(role_after)
    assert role_phase_tv>.5
    for role in phase_model.lexical_roles:
        if role.origin!="explicit_function_grammar":
            role.spectrum[:]=0
    assert phase_model.generate("Was kannst du?")["text"]==""
    named = model.generate("Ich heiße Xyrländo.")
    assert model.generate("Wie heiße ich?",context=named["state"])["text"]==named["text"]
    assert model.generate("Wie heiße ich?",context={})["text"]==""
    changed_records = json.loads(json.dumps(records))
    for row in changed_records:
        row["text"] = row["text"].replace("heißt FreqAI","heißt Kymara")
        row["text"] = row["text"].replace("Lumor hat eine Frequenz von 19","Lumor hat eine Frequenz von 73")
        row["text"] = row["text"].replace("Tavor hat eine Frequenz von 73","Tavor hat eine Frequenz von 19")
    changed = UnpairedWaveModel(changed_records)
    changed_responses = {prompt:changed.generate(prompt)["text"] for prompt in
                         ["Wie heißt du?","Welche Frequenz hat das Signal Lumor?",
                          "Welches Signal hat eine Frequenz von 19 Hertz?"]}
    assert "Kymara" in changed_responses["Wie heißt du?"]
    assert "73 Hertz" in changed_responses["Welche Frequenz hat das Signal Lumor?"]
    assert "Tavor" in changed_responses["Welches Signal hat eine Frequenz von 19 Hertz?"]
    signature_after = model.model_signature()
    assert signature_before["coefficient_sha256"]==signature_after["coefficient_sha256"]
    assert signature_before["declaration_role_sha256"]==signature_after["declaration_role_sha256"]
    rebuilt = UnpairedWaveModel(records).model_signature()
    assert rebuilt["coefficient_sha256"]==signature_before["coefficient_sha256"]
    assert rebuilt["declaration_role_sha256"]==signature_before["declaration_role_sha256"]
    for role in model.lexical_roles:
        role.spectrum[:]=0
    for spectrum in model.transition_spectra.values():
        spectrum[:]=0
    zero_data = {prompt:model.generate(prompt)["text"] for prompt in ["Hallo","Wie heißt du?","Was ist Befinden?"]}
    assert not any(zero_data.values())
    report = {"created_at":datetime.now(timezone.utc).isoformat(),
              "scope":"numerical controls on 45 authored declarations and 3 openly generated facts; not a language holdout",
              "code_sha256":hashlib.sha256((ROOT/"freqai/unpaired.py").read_bytes()).hexdigest(),
              "conversation_facts_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "all_inputs_unpaired":all(not set(row)&{"prompt","question","answer","messages"} for row in records),
              "optimizer_steps":0,"fitted_mixing_gains":0,"checked_prefix_time_states":checked_states,
              "fft_direct_max_probability_error_after_rounding":maximum_error,
              "prompt_phase_total_variation":phase_tv,"phase_changes_selected_symbol":True,
              "data_role_phase_total_variation":role_phase_tv,"data_role_phase_changes_selected_symbol":True,
              "zero_roles_refuse_with_prose_transition_fields_intact":True,
              "unknown_name_byte_roundtrip":"Xyrländo","unknown_name_isolated_between_sessions":True,
              "zero_prompt_refuses":zero_prompt_refuses,"zero_data_responses":zero_data,
              "same_inputs_rebuild_identical_coefficients":True,"queries_do_not_modify_corpus_coefficients":True,
              "no_global_session_roles_retained":not any(role.origin=="session_prompt_only" for role in model.lexical_roles),
              "counterfactual_responses":changed_responses,"model_signature":signature_before,
              "examples":rows,
              "limitations":["Semantic role binding and speech-act grammar are explicit human-authored algorithms.",
                  "Passing numerical controls does not establish general conversation ability or factual coverage.",
                  "Unknown prompt words consume UTF-8 byte symbols; token budgets are not always word budgets.",
                  "Declarations retain ordered predicate arguments; grammar composes responses without stored full-answer candidates."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"checked_states":checked_states,"phase_tv":phase_tv,
                      "fft_direct_max_error":maximum_error},ensure_ascii=False))


if __name__=="__main__":
    main()
