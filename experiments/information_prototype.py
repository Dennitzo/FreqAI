"""Prose-only scaling and numerical audit; never reads evaluation questions."""
from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np

from freqai.information import InformationWaveModel, tokenize
from experiments.benchmark_generative_storage import peak_working_set


def numerical_audit():
    # Reuse only the independently authored unit-test Toy corpus, never any
    # development or held-out corpus/questions.
    spec=importlib.util.spec_from_file_location('information_toy_tests',ROOT/'tests/test_information.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model=InformationWaveModel(module.PROSE)
    rebuilt=InformationWaveModel(module.PROSE)
    field,_=model.prompt_field('Was ist Frequenz?')
    rng=np.random.default_rng(893721)
    rows=[]; data_changes=[]
    for intervention in ('observed','complex_phase'):
        if intervention=='complex_phase':
            before=model.next_distribution(['die','frequenz','ist'],field)[0]
            for spectrum in model.transition_spectra.values():
                spectrum.local_spectrum *= np.exp(2j*np.pi*rng.random(len(spectrum.local_spectrum)))
            after=model.next_distribution(['die','frequenz','ist'],field)[0]
            data_changes.append(float(np.abs(before-after).sum()/2))
        for prefix in ([],['die'],['frequenz','ist']):
            for time_s in (0,73.125,1e9):
                for phase in (0,0.8,np.pi):
                    controls={'time_s':time_s,'phase_error':phase}
                    try:
                        operator,_=model.next_distribution(prefix,field,**controls)
                    except ValueError as error:
                        try:
                            model.next_distribution(prefix,field,method='direct',**controls)
                            raise AssertionError('Direct calculation failed to cancel with operator')
                        except ValueError:
                            pass
                        rows.append({'intervention':intervention,'prefix':prefix,**controls,
                                     'destructive_zero_refusal':True,'error':str(error)})
                        continue
                    direct,_=model.next_distribution(prefix,field,method='direct',**controls)
                    timeless,_=model.next_distribution(prefix,field,time_s=0,phase_error=phase)
                    rows.append({'intervention':intervention,'prefix':prefix,**controls,'destructive_zero_refusal':False,
                                 'operator_direct_error':float(np.max(np.abs(operator-direct))),
                                 'time_error':float(np.max(np.abs(operator-timeless)))})
    prompt_null_refused=False
    try:
        model.next_distribution([],field*0)
    except ValueError:
        prompt_null_refused=True
    conditioned,_=rebuilt.next_distribution(['die','frequenz','ist'],field)
    prefix_null,_=rebuilt.next_distribution(['die','frequenz','ist'],field,prefix_enabled=False)
    constructive,_=rebuilt.next_distribution(['die','frequenz','ist'],field,phase_error=0)
    alternative,_=rebuilt.next_distribution(['die','frequenz','ist'],field,phase_error=0.8)
    for spectrum in model.transition_spectra.values():
        spectrum.local_spectrum[:]=0
    data_null_refused=False
    try:
        model.next_distribution([],field)
    except ValueError:
        data_null_refused=True
    toy_texts=['Rote Katzen schlafen.','Blaue Hunde laufen.','Blaue Katzen laufen.']
    toy=InformationWaveModel([{'title':'Tiere','text':text} for text in toy_texts],order=1)
    causal=InformationWaveModel([{'title':'Tiere','text':text} for text in toy_texts],order=1)
    causal_field,_=causal.prompt_field('Tiere')
    before,causal_trace=causal.next_distribution([],causal_field)
    for address in causal_trace['field_addresses']:
        spectrum=causal.transition_spectra[(address['group'],tuple(address['prefix']))]
        spectrum.local_spectrum *= np.exp(2j*np.pi*rng.random(len(spectrum.local_spectrum)))
    after,_=causal.next_distribution([],causal_field)
    multiple_mode_data_change=float(np.abs(before-after).sum()/2)
    phase_field,_=toy.prompt_field('Tiere')
    phase_field.token_amplitudes[toy.index['rote']]=1
    phase_constructive,_=toy.next_distribution([],phase_field,phase_error=0)
    phase_destructive,_=toy.next_distribution([],phase_field,phase_error=np.pi)
    known={tuple(tokenize(text)) for text in toy_texts}
    novelty=[]
    for seed in range(12):
        result=toy.generate('Erzähle über Tiere.',decoding='sample',seed=seed)
        if result['ended'] and tuple(result['tokens']) not in known:
            novelty.append({'seed':seed,'text':result['text'],'tokens':result['tokens']})
    signatures=[InformationWaveModel(module.PROSE).model_signature() for _ in range(2)]
    finite=[row for row in rows if not row['destructive_zero_refusal']]
    report={'scope':'Independent unit-test Toy prose only; no evaluation questions.',
        'core_sha256':hashlib.sha256((ROOT/'freqai/information.py').read_bytes()).hexdigest(),
        'checked_states':len(rows),'finite_probability_states':len(finite),
        'destructive_zero_refusals':len(rows)-len(finite),
        'operator_direct_max_error':max(row['operator_direct_error'] for row in finite),
        'time_max_error':max(row['time_error'] for row in finite),
        'data_phase_total_variations':data_changes,'prompt_null_refused':prompt_null_refused,
        'actual_complex_data_null_refused':data_null_refused,
        'prefix_null_total_variation':float(np.abs(conditioned-prefix_null).sum()/2),
        'relative_phase_total_variation_at_example_prefix':float(np.abs(constructive-alternative).sum()/2),
        'multiple_mode_data_phase_total_variation':multiple_mode_data_change,
        'multiple_mode_relative_phase_total_variation':float(np.abs(phase_constructive-phase_destructive).sum()/2),
        'multiple_mode_relative_phase_changes_top_token':bool(np.argmax(phase_constructive)!=np.argmax(phase_destructive)),
        'coefficient_rebuild_equal':signatures[0]['coefficient_sha256']==signatures[1]['coefficient_sha256'],
        'coefficient_sha256':signatures[0]['coefficient_sha256'],
        'new_sentence_examples':novelty,'rows':rows,
        'interpretation':['Coefficients are numerical weights calculated from observed counts; no gradient optimization.',
            'No dataset-specific gains or old dialogue mixture coefficients are used.',
            'Numerical dependence and new Toy combinations do not prove general factual understanding.']}
    destination=ROOT/'results/information/numerical_audit.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('rows','new_sentence_examples')},ensure_ascii=True))
    print(json.dumps(novelty,ensure_ascii=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',type=Path,default=ROOT/'memory/imports/information_wikipedia_20000.jsonl')
    parser.add_argument('--limit',type=int,default=1000)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.audit_only:
        numerical_audit()
        return
    raw=args.corpus.read_bytes()
    records=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()][:args.limit]
    start=time.perf_counter()
    model=InformationWaveModel(records)
    build_seconds=time.perf_counter()-start
    report={'created_utc':datetime.now(timezone.utc).isoformat(), 'corpus_path':str(args.corpus),
        'corpus_sha256':hashlib.sha256(raw).hexdigest(),'requested_records':args.limit,
        'core_sha256':hashlib.sha256((ROOT/'freqai/information.py').read_bytes()).hexdigest(),
        'build_seconds':build_seconds,'storage':model.storage_stats(),'probes':[],
        'scope':'Numerical/scaling probes authored in this script; no evaluation/holdout data is read.'}
    print(json.dumps({'build_seconds':build_seconds,**model.storage_stats()}),flush=True)
    # These explicit, previously discussed user/Toy prompts are not an unseen
    # quality evaluation. They expose concrete behavior and scaling only.
    for question in ('Was ist eine Frequenz?', 'In welcher Einheit wird die Frequenz gemessen?',
                     'Wie hängen Frequenz und Periodendauer zusammen?'):
        start=time.perf_counter()
        output=model.generate(question,max_tokens=64)
        seconds=time.perf_counter()-start
        row={'prompt':question,'answer':output['text'],'tokens':output['tokens'],'ended':output['ended'],
             'reason':output['reason'],'seconds':seconds,'analysis':output['analysis'],'numerics':[]}
        if output['tokens']:
            field,_=model.prompt_field(question)
            for prefix in ([],output['tokens'][:3],output['tokens'][:8]):
                actual,trace=model.next_distribution(prefix,field,time_s=86400.125)
                direct,_=model.next_distribution(prefix,field,time_s=86400.125,method='direct')
                timeless,_=model.next_distribution(prefix,field,time_s=0)
                null_prefix,_=model.next_distribution(prefix,field,prefix_enabled=False)
                row['numerics'].append({'prefix':prefix,'operator_direct_max_error':float(np.max(np.abs(actual-direct))),
                    'time_max_error':float(np.max(np.abs(actual-timeless))),
                    'prefix_null_total_variation':float(np.abs(actual-null_prefix).sum()/2),
                    'active_tokens':len(trace['active_token_indices']),'carrier_size':trace['carrier_size']})
        report['probes'].append(row)
        print(json.dumps(row,ensure_ascii=True),flush=True)
    report['peak_process_working_set_bytes']=peak_working_set()
    report['signature']=model.model_signature()
    destination=ROOT/f'results/information/prototype_{args.limit}.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(destination),flush=True)


if __name__=='__main__':
    main()
