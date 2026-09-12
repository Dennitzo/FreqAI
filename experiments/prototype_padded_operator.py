"""Isolated operator-only FFT carrier padding; never modifies production code.

The imported global fields and symbol frequencies keep their V-dimensional
meaning. Only interference is evaluated on N >= V carrier modes, by appending
zero-amplitude modes before every matching Fourier transform. IFFT readout uses
the original first V symbol modes; no spectrum-bin padding or word pruning.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import sys
import textwrap
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import fft as scipy_fft
from scipy.fft import next_fast_len

import freqai.generative as core
from freqai.generative import GenerativeWaveModel, PromptField, tokenize


def prototype_method():
    original = textwrap.dedent(inspect.getsource(GenerativeWaveModel.next_distribution))
    changes = [(
        '    if self.storage == "compact":\n        base_spectrum = np.fft.fft(base_modes, norm="ortho")\n',
        '    carrier_size = next_fast_len(self.size) if method == "operator" else self.size\n'
        '    if self.storage == "compact":\n'
        '        base_spectrum = np.fft.fft(base_modes, n=carrier_size, norm="ortho")\n'
        '    elif carrier_size != self.size:\n'
        '        base_spectrum = np.fft.fft(np.fft.ifft(base_spectrum, norm="ortho"),\n'
        '                                   n=carrier_size, norm="ortho")\n'), (
        '    phase_spectrum = np.fft.fft(phases, norm="ortho")\n',
        '    phase_spectrum = np.fft.fft(phases, n=carrier_size, norm="ortho")\n'), (
        '        prompt_spectrum = np.fft.fft(np.asarray(prompt_field), norm="ortho")\n',
        '        prompt_spectrum = np.fft.fft(np.asarray(prompt_field), n=carrier_size, norm="ortho")\n'), (
        '    scores = np.abs(amplitudes)**2\n',
        '    scores = np.abs(amplitudes[:self.size])**2\n'), (
        '             "decoder": method, "support_size": int(support.sum()),\n',
        '             "decoder": method, "support_size": int(support.sum()), "carrier_bins": carrier_size,\n')]
    modified = original
    for before, after in changes:
        if modified.count(before) != 1:
            raise RuntimeError("Core drift: inspect the candidate before applying another replacement")
        modified = modified.replace(before, after)
    namespace = {**vars(core), "next_fast_len": next_fast_len}
    exec(compile(modified, str(Path(__file__).resolve())+":candidate", "exec"), namespace)
    return namespace["next_distribution"], original, modified


CANDIDATE, ORIGINAL_SOURCE, CANDIDATE_SOURCE = prototype_method()


class PaddedCarrierWaveModel(GenerativeWaveModel):
    next_distribution = CANDIDATE


def operator_checks():
    rng = np.random.default_rng(6829031)
    rows = []
    for storage in ("compact", "dense"):
        model = PaddedCarrierWaveModel([], order=2, storage=storage, conditioned_pairs=[
            {"prompt": "Rote Tiere", "text": "Rote Katzen schlafen im Garten."},
            {"prompt": "Blaue Tiere", "text": "Blaue Hunde laufen am Fluss."},
            {"prompt": "Rote Tiere", "text": "Rote Hunde laufen durch Blumenwiesen."},
        ])
        # These are deliberately nontrivial complex data mutations, not just
        # original nonnegative corpus fields, to check phase causality too.
        for intervention in ("original", "global_complex_phase"):
            if intervention != "original":
                phase = np.exp(2j*np.pi*rng.random(model.size))
                for bank in (model.transition_spectra, model.conditioned_spectra):
                    for key in bank:
                        bank[key] = np.asarray(bank[key])*phase
            field, _ = model.prompt_field("Rote Tiere")
            variants = (field, field*0, PromptField(np.zeros(model.size), field.feature_spectrum))
            for field_variant, numerical_field in enumerate(variants):
                for prefix in ([], ["rote"], ["im", "garten"], ["unbekannt"]):
                    for time_s, relative_phase in ((0, 0), (123.456, 0.8), (1e9, np.pi)):
                        original, _ = GenerativeWaveModel.next_distribution(
                            model, prefix, numerical_field, time_s=time_s, phase_error=relative_phase)
                        actual, trace = model.next_distribution(prefix, numerical_field,
                            time_s=time_s, phase_error=relative_phase)
                        direct, _ = model.next_distribution(prefix, numerical_field, method="direct",
                            time_s=time_s, phase_error=relative_phase)
                        rows.append({"storage":storage, "intervention":intervention,
                            "field_variant":field_variant, "prefix":prefix, "time_s":time_s,
                            "relative_phase":float(relative_phase), "vocabulary_bins":model.size,
                            "carrier_bins":trace["carrier_bins"],
                            "original_max_error":float(np.max(np.abs(actual-original))),
                            "direct_max_error":float(np.max(np.abs(actual-direct))),
                            "same_top_token":bool(np.argmax(actual)==np.argmax(original))})
        for bank in (model.transition_spectra, model.conditioned_spectra):
            for value in bank.values():
                value[:] = 0
        try:
            model.next_distribution([], field)
            raise AssertionError("Zero data wave unexpectedly produced token probabilities")
        except ValueError:
            pass
    assert max(row["original_max_error"] for row in rows) < 1.01e-12
    assert max(row["direct_max_error"] for row in rows) < 1.01e-12
    assert all(row["same_top_token"] for row in rows)
    return {"checked_states":len(rows), "zero_memory_refusals":2,
            "maximum_original_error":max(row["original_max_error"] for row in rows),
            "maximum_direct_error":max(row["direct_max_error"] for row in rows), "rows":rows}


def fft_backends():
    rng = np.random.default_rng(802197)
    result = []
    for size in (44614, 44615, 44800):
        values = rng.standard_normal(size)+1j*rng.standard_normal(size)
        timings = {}
        for label, fft in (("numpy",np.fft.fft), ("scipy",scipy_fft.fft)):
            fft(values)
            samples=[]
            for _ in range(12):
                start=time.perf_counter(); fft(values); samples.append(time.perf_counter()-start)
            timings[label]=float(np.median(samples))
        result.append({"bins":size,"fft_median_seconds":timings,
            "numpy_to_scipy_speedup":timings['numpy']/timings['scipy'],
            "maximum_fft_error":float(np.max(np.abs(np.fft.fft(values)-scipy_fft.fft(values))))})
    return result


def full_corpus_benchmark():
    from freqai.generation_runtime import corpus_annotations, corpus_pairs, GENERATION_CONFIGURATION, MODEL_CONFIGURATION
    from freqai.memory import Document
    # Read only text documents, never conversation history or evaluation data.
    database = ROOT / "memory/memory.sqlite3"
    with sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        connection.row_factory=sqlite3.Row
        documents=[Document(**dict(row)) for row in connection.execute(
            "SELECT id,text,source,prompt FROM documents ORDER BY sequence")]
    documents_by_id={document.id:document for document in documents}
    sources=[]
    for relative in ("memory/imports/public_dialogue_oasst2_de.jsonl", "memory/imports/public_knowledge_germanquad_de.jsonl"):
        path=ROOT / relative; raw=path.read_bytes()
        records=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        sources.append({"path":relative,"sha256":hashlib.sha256(raw).hexdigest(),"records":len(records)})
        for row in records:
            document=Document(**{key:row[key] for key in ('id','text','source','prompt')})
            documents_by_id[document.id]=document
    documents=list(documents_by_id.values())
    start=time.perf_counter()
    model=PaddedCarrierWaveModel([document.text for document in documents],
        conditioned_pairs=corpus_pairs(documents), annotated_texts=corpus_annotations(documents), **MODEL_CONFIGURATION)
    build_seconds=time.perf_counter()-start
    print(json.dumps({"built_records":len(documents),"tokens":model.size,
        "features":len(model.feature_vocabulary),"build_seconds":build_seconds}),flush=True)
    # Fixed seen-corpus probes are sufficient for a pure performance regression.
    # They are not new evaluation/holdout prompts or a quality optimization set.
    probes=[document for document in documents if document.prompt][:3]
    numerical=[]
    for doc in probes:
        field,_=model.prompt_field(doc.prompt)
        for prefix in ([],tokenize(doc.text)[:3]):
            for time_s,phase in ((0,0),(86400.125,np.pi)):
                start=time.perf_counter()
                expected,_=GenerativeWaveModel.next_distribution(model,prefix,field,time_s=time_s,phase_error=phase)
                baseline_seconds=time.perf_counter()-start
                start=time.perf_counter()
                actual,trace=CANDIDATE(model,prefix,field,time_s=time_s,phase_error=phase)
                candidate_seconds=time.perf_counter()-start
                direct,_=CANDIDATE(model,prefix,field,method='direct',time_s=time_s,phase_error=phase)
                numerical.append({"source_id":doc.id,"prefix":prefix,"time_s":time_s,
                    "phase":float(phase),"baseline_seconds":baseline_seconds,"candidate_seconds":candidate_seconds,
                    "original_max_error":float(np.max(np.abs(expected-actual))),
                    "direct_max_error":float(np.max(np.abs(direct-actual))),"carrier_bins":trace['carrier_bins']})
    generations=[]
    for doc in probes:
        outputs={}; timings={}
        for label,method in (("baseline",GenerativeWaveModel.next_distribution),("padded",CANDIDATE)):
            model.next_distribution=MethodType(method,model)
            start=time.perf_counter()
            outputs[label]=model.generate(doc.prompt,**GENERATION_CONFIGURATION,max_tokens=40,seed=17)
            timings[label]=time.perf_counter()-start
        generations.append({"source_id":doc.id,"seconds":timings,
            "speedup":timings['baseline']/timings['padded'],
            "tokens_equal":outputs['baseline']['tokens']==outputs['padded']['tokens'],
            "output":outputs['padded']['text'],"token_count":len(outputs['padded']['tokens'])})
        print(json.dumps(generations[-1],ensure_ascii=True),flush=True)
    return {"source_documents":len(documents),"sources":sources,"vocabulary_bins":model.size,
        "feature_bins":len(model.feature_vocabulary),"build_seconds":build_seconds,
        "numerical":numerical,"generations":generations,
        "median_single_step_speedup":float(np.median([row['baseline_seconds']/row['candidate_seconds'] for row in numerical])),
        "median_generation_speedup":float(np.median([row['speedup'] for row in generations])),
        "maximum_original_error":max(row['original_max_error'] for row in numerical),
        "maximum_direct_error":max(row['direct_max_error'] for row in numerical),
        "all_complete_outputs_equal":all(row['tokens_equal'] for row in generations)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-corpus',action='store_true')
    args=parser.parse_args()
    output=ROOT/'results/public_corpus/performance'
    output.mkdir(parents=True,exist_ok=True)
    (output/'operator_padding_method.py').write_text(CANDIDATE_SOURCE,encoding='utf-8')
    (output/'operator_padding_method.diff').write_text(''.join(difflib.unified_diff(
        ORIGINAL_SOURCE.splitlines(keepends=True),CANDIDATE_SOURCE.splitlines(keepends=True),
        fromfile='original next_distribution',tofile='prototype next_distribution')),encoding='utf-8')
    report={'production_core_changed':False,'prototype':'experiments/prototype_padded_operator.py',
        'core_sha256':hashlib.sha256((ROOT/'freqai/generative.py').read_bytes()).hexdigest(),
        'mechanism':'Zero-amplitude padding of token modes before runtime FFT; imported spectra, dictionaries and stable symbol frequencies unchanged.',
        'unchanged_reference':'method=direct retains the original V-dimensional computation; feature channel unchanged.',
        'fft_backends':fft_backends(),'small_complex_checks':operator_checks()}
    path=output/'operator_padding_report.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='small_complex_checks'},ensure_ascii=True),flush=True)
    print(json.dumps({key:value for key,value in report['small_complex_checks'].items() if key!='rows'}),flush=True)
    if args.full_corpus:
        report['full_corpus']=full_corpus_benchmark()
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({key:value for key,value in report['full_corpus'].items() if key not in ('numerical','generations','sources')}),flush=True)


if __name__=='__main__':
    main()
