"""Calibrate numerical reproducibility on fixed input without viewing heldout outcomes."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    proof=ROOT/'runs/persisted_autotune_probe/result.json';deadline=time.monotonic()+3500
    while not proof.exists():
        if time.monotonic()>deadline:raise TimeoutError('Ref numerical probe unfinished')
        time.sleep(5)
    assert json.loads(proof.read_text())['passed']
    out=ROOT/'runs/inference_autotune_calibration';assert not out.exists();out.mkdir()
    cache_dir=ROOT/'cache/inference_autotune';assert not cache_dir.exists();cache_dir.mkdir()
    refcache=cache_dir/'ref_N10.pb';shutil.copy2(ROOT/'runs/persisted_autotune_probe/ref_autotune.pb',refcache)
    state=json.loads((ROOT/'evidence/full_experiment_state.json').read_text());assert set(state['models'])=={'B1','A1'}
    for path in (ROOT/'runs').glob('*/episodes/**/run.json'):
        assert json.loads(path.read_text()).get('settings',{}).get('plus_split')!='heldout'
    modes=[('lora10_capture','B1',10,True),('lora10_reload','B1',10,False),('A1_lora10_reload','A1',10,False),('lora3_capture','A1',3,True),('lora3_reload','A1',3,False)]
    records=[]
    for name,model,n,capture in modes:
        cache=cache_dir/f'lora_N{n}.pb'
        flags='--xla_gpu_dump_autotune_results_to='+str(cache) if capture else '--xla_gpu_load_autotune_results_from='+str(cache)+' --xla_gpu_require_complete_aot_autotune_results=true'
        env=dict(os.environ,OPENPI_AUG='default',XLA_PYTHON_CLIENT_MEM_FRACTION='.8',XLA_FLAGS=flags)
        result_path=out/(name+'.json')
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([str(ROOT/'third_party/openpi/.venv/bin/python'),str(ROOT/'scripts/checkpoint_action_sanity.py'),'--checkpoint',state['models'][model]['checkpoint'],'--observation',str(ROOT/'runs/policy_sanity_input/observation.npz'),'--out',str(result_path),'--num-steps',str(n)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        result=json.loads(result_path.read_text());assert result['passed'] and cache.is_file()
        records.append({'name':name,'model':model,'num_steps':n,'output_sha256':[r['sha256'] for r in result['records']],'cache_sha256':sha(cache)})
        (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    by_name={r['name']:r for r in records}
    assert by_name['lora10_capture']['output_sha256']==by_name['lora10_reload']['output_sha256']
    assert by_name['lora3_capture']['output_sha256']==by_name['lora3_reload']['output_sha256']
    registration={'status':'frozen','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'strategy':'Persist autotuning once per policy graph; require complete cached decisions for every scored server. No weight, augmentation, R/N or task changes.',
        'heldout_outcomes_accessed':False,'ref_source_and_restart_proof_sha256':sha(proof),'calibration_records':records,
        'caches':{key:{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for key,path in {
            'pi05_libero_N10':refcache,'pi05_libero_lora_ft_N10':cache_dir/'lora_N10.pb','pi05_libero_lora_ft_N3':cache_dir/'lora_N3.pb'}.items()},
        'original_exact_episode_replay':'Failed, retained as a diagnostic. Original389/400 and pre-fix387/400 are separate historical evaluations; neither pooled nor relabeled as a method gain.',
        'final_reference_rule':'Run a fresh Ref400 with fixed kernel decisions; use the common final execution settings for Ref/B1/A1. Preserve all old results and register runtime amendment before heldout.',
        'claim_scope':'Fixed-input byte equality across restarts/source variants plus cached-kernel enforcement; does not prove universal GPU determinism or training-seed robustness.',
        'reference':'https://openxla.org/xla/determinism'}
    (ROOT/'configs/inference_reproducibility_registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps({'passed':True,'registration_sha256':sha(ROOT/'configs/inference_reproducibility_registration.json')},indent=2)+'\n');print('Numerical inference calibration passed',flush=True)


if __name__=='__main__':main()
