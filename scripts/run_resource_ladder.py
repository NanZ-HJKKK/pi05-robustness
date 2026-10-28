"""Select a common feasible batch using measured100step control/augmentation probes."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'third_party/openpi/.venv/bin/python'
INDEX=ROOT/'evidence/resource_ladder.json'


def call(command,environment,log):
    with log.open('w') as stream:
        subprocess.run([str(x) for x in command],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)


def validate_and_release(directory):
    result=json.loads((directory/'result.json').read_text());assert result['success']
    checkpoint=result['checkpoint_expected'];env=dict(os.environ,JAX_PLATFORMS='cpu')
    call([PYTHON,ROOT/'scripts/verify_checkpoint_reload.py',checkpoint,'--audit-dir',directory,'--out',directory/'reload_cpu.json'],env,directory/'reload_cpu.log')
    env=dict(os.environ,OPENPI_AUG='default',XLA_PYTHON_CLIENT_MEM_FRACTION='.9')
    env.pop('JAX_PLATFORMS',None)
    call([PYTHON,ROOT/'scripts/checkpoint_action_sanity.py','--checkpoint',checkpoint,'--observation',ROOT/'runs/policy_sanity_input/observation.npz','--out',directory/'actions_sanity.json'],env,directory/'actions_sanity.log')
    call([sys.executable,ROOT/'scripts/release_resource_checkpoint.py','--run',directory],dict(os.environ),directory/'release.log')
    return result


def run_probe(batch,augmentation,tag):
    name=f'memtest_{augmentation}_b{batch}_{tag}';directory=ROOT/'runs'/name
    env=dict(os.environ,HF_LEROBOT_HOME=str(ROOT/'cache/training'))
    command=[sys.executable,ROOT/'scripts/run_training.py','--openpi',ROOT/'third_party/openpi','--stage','memtest','--batch',batch,'--steps',100,'--max-minutes',30,'--experiment-name',name,'--probe-augmentation',augmentation,'--dataset-manifest',ROOT/'runs/dataset_download/verified_manifest.json','--out',directory]
    with (ROOT/'runs'/(name+'_supervisor.log')).open('w') as stream:
        completed=subprocess.run([str(x) for x in command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1950)
    result=json.loads((directory/'result.json').read_text())
    record={'batch':batch,'augmentation':augmentation,'run':str(directory),'success':result['success'],'exit_code':completed.returncode,'peak_allocator_bytes_in_use':result['peak_allocator_bytes_in_use'],'median_step_seconds':result['median_step_seconds_excluding_first_5']}
    if result['success']:validate_and_release(directory)
    return record


def main():
    assert not INDEX.exists(),'Resource selection already registered'
    original=ROOT/'runs/memtest_default_b4_20261027T034400Z'
    deadline=time.monotonic()+1800
    while not (original/'result.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Batch4 did not finish')
        time.sleep(5)
    validate_and_release(original)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'peak_limit_GiB':22.5,'candidates':[16,8,4],'records':[],
        'rule':'Choose largest candidate for which both default and registered augmentation100step training pass, including update/freeze, checkpoint CPU reload and finite deterministic actions; peak allocator<=22.5GiB. No task outcome used.'}
    INDEX.write_text(json.dumps(report,indent=2)+'\n')
    for batch in report['candidates']:
        accepted=[]
        for aug in ['default','view_wrist_drop']:
            record=run_probe(batch,aug,tag);report['records'].append(record);INDEX.write_text(json.dumps(report,indent=2)+'\n')
            if not record['success'] or record['peak_allocator_bytes_in_use']>22.5*1024**3:break
            accepted.append(record)
        if len(accepted)==2:
            a,b=[Path(r['run'])/'first_batches.jsonl' for r in accepted]
            assert a.read_bytes()==b.read_bytes(),'Control/augmentation first real batches differ'
            report.update(status='complete',selected_batch=batch,first10_input_batches_identical=True,ended_at_utc=datetime.now(timezone.utc).isoformat())
            INDEX.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True);return
    raise RuntimeError('No common feasible batch; inspect recorded failures')


if __name__=='__main__':main()
