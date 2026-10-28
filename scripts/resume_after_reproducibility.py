"""Resume only evaluations after verified kernel reproducibility; never retrain models."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import time
import run_full_experiment as study

ROOT=study.ROOT


def verify_reference_inputs(out):
    old={r['pair_id']:r for r in map(json.loads,(ROOT/'runs/ref_clean400_20261026T235719Z/clean400/episodes.jsonl').read_text().splitlines())}
    new={r['pair_id']:r for p in (out/'episodes').glob('*/episodes.jsonl') for r in map(json.loads,p.read_text().splitlines())}
    assert len(new)==400 and set(old)==set(new)
    fields=['initial_state_sha256','settled_sim_state_sha256','initial_main_rgb_sha256','initial_wrist_rgb_sha256','prompt','flow_noise_identity','seed']
    assert all(old[k][field]==new[k][field] for k in old for field in fields)
    report={'inputs_matched':True,'episodes':400,'original_successes':sum(r['success'] for r in old.values()),
        'fixed_kernel_successes':sum(r['success'] for r in new.values()),
        'outcome_disagreements':sum(old[k]['success']!=new[k]['success'] for k in old),
        'scope':'Original default-autotune and fixed-kernel runs are separate evaluations. Inputs match; historical bitwise trajectory/outcome equality is not claimed and failed original bridge remains preserved.'}
    (out/'historical_reference_comparison.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    signal.signal(signal.SIGTERM,study.interrupt);signal.signal(signal.SIGINT,study.interrupt)
    deadline=time.monotonic()+4000;proof=ROOT/'runs/inference_autotune_calibration/result.json'
    while not proof.exists():
        if time.monotonic()>deadline:raise TimeoutError('Calibration did not complete')
        time.sleep(5)
    assert study.read(proof)['passed']
    state=study.read(study.STATE);assert state['status']=='stopped_on_error' and 'Ref bridge changed' in state['error']
    assert set(state['models'])=={'B1','A1'} and len(state['evaluations'])==1
    for model,r in state['models'].items():
        run=Path(r['run']);assert study.read(run/'result.json')['success']
        assert study.read(run/'reload_cpu.json')['passed'] and study.read(run/'actions_sanity.json')['passed']
    plan=study.read(ROOT/'configs/training_plan.frozen.json')
    assert study.read(ROOT/'third_party/openpi/robustness_patch_manifest.json')==plan['source_hashes']
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');history=ROOT/'runs'/('evaluation_restart_'+tag);history.mkdir()
    shutil.copy2(study.STATE,history/'state_before_restart.json')
    amendment={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'before_any_final_heldout_outcomes':True,
        'training_changed':False,'reason':'Controlled fixed-input tests reproduce cross-compilation drift and eliminate it by persisting autotuning decisions; old/current source variants then match native action bytes.',
        'old_exact_bridge_passed':False,'old_bridge_preserved':str(Path(state['evaluations'][0]['out'])/'baseline_equivalence.json'),
        'new_execution_rule':'All scored servers require complete pre-calibrated autotune decisions, selected without task outcomes. Fresh common Ref400, followed by original B1/A1 clean/heldout and five efficiency jobs. Initial identities still checked exactly.',
        'inference_registration_sha256':study.sha(ROOT/'configs/inference_reproducibility_registration.json'),
        'original_analysis_endpoints_unchanged':True,'historical_reference_runs_never_pooled':True}
    (ROOT/'configs/evaluation_runtime_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
    state['validation_runs']=[{'out':r['out'],'reason':'Failed original exact replay; preserve, exclude from final common comparison'} for r in state['evaluations']]
    for path in ['runs/ref_repeatability_diagnosis','runs/persisted_autotune_probe','runs/inference_autotune_calibration',str(history.relative_to(ROOT))]:
        state['validation_runs'].append({'out':str(ROOT/path),'reason':'Numerical reproducibility diagnosis/calibration or restart history'})
    state.update(training_run_tag=state['tag'],tag=tag,status='running',evaluations=[],runtime_amendment_sha256=study.sha(ROOT/'configs/evaluation_runtime_amendment.json'),inference_reproducibility_registration_sha256=amendment['inference_registration_sha256'])
    state.pop('error',None);study.write_state(state,'resumed_evaluations_with_fixed_kernels');os.environ['OPENPI_FIXED_AUTOTUNE']='1'
    try:
        for model in ['Ref','B1','A1']:
            clean=study.evaluate(model,'clean',state)
            if model=='Ref':verify_reference_inputs(clean)
            study.evaluate(model,'plus',state)
        for r,n in [(1,10),(5,10),(10,10),(5,3),(10,3)]:study.evaluate('A1','clean',state,r,n,serial=True)
        state['status']='all_jobs_complete_pending_independent_acceptance';study.write_state(state,'analysis_ready')
    except Exception as exc:
        state.update(status='stopped_on_error',error=type(exc).__name__+': '+str(exc));study.write_state(state,'inspect_error');raise


if __name__=='__main__':main()
