"""Finish the registered study on the existing GPU; never change cloud power state."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'third_party/openpi/.venv/bin/python'
STATE=ROOT/'evidence/full_experiment_state.json'
ACTIVE=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):
    for attempt in range(10):
        try:return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.1)


def write_state(state,phase):
    state.update(phase=phase,updated_at_utc=datetime.now(timezone.utc).isoformat())
    temporary=STATE.with_suffix('.writing');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(STATE);print(phase,flush=True)


def interrupt(signum,frame):raise TimeoutError('Full experiment interrupted by signal '+str(signum))


def call(command,log,timeout,env=None):
    global ACTIVE
    with Path(log).open('w') as stream:
        ACTIVE=subprocess.Popen([str(x) for x in command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=ACTIVE.wait(timeout=timeout)
            if code:raise RuntimeError('Child failed: '+str(log)+' exit='+str(code))
        finally:
            if ACTIVE.poll() is None:
                os.killpg(ACTIVE.pid,signal.SIGTERM)
                try:ACTIVE.wait(timeout=90)
                except subprocess.TimeoutExpired:os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait(timeout=10)
            ACTIVE=None


def freeze_plan(resource):
    plan_path=ROOT/'configs/training_plan.frozen.json';assert not plan_path.exists()
    assert resource['status']=='complete' and resource['first10_input_batches_identical']
    registration=read(ROOT/'configs/augmentation_registration.json')
    assert sha(ROOT/'src/pi05_robustness/augmentation.py')==registration['helper_sha256']
    assert sha(ROOT/'runs/ref_diagnosis/diagnosis.json')==registration['diagnosis_report_sha256']
    assert sha(ROOT/'runs/ref_camera_failure_review/review.json')==registration['failure_review_sha256']
    assert read(ROOT/'runs/registered_augmentation_preview/result.json')['passed']
    assert read(ROOT/'runs/ref_plus_diagnostic_20261027T015638Z/result.json')['status']=='complete'
    for path in (ROOT/'runs').rglob('run.json'):
        assert read(path).get('settings',{}).get('plus_split')!='heldout','Heldout already consumed before freeze'
    plan=read(ROOT/'configs/training_plan.draft.json')
    plan.update(status='frozen',frozen_at_utc=datetime.now(timezone.utc).isoformat(),batch_size=resource['selected_batch'],
        num_train_steps=5000,augmentation='view_wrist_drop',augmentation_status='registered_after_diagnostic_before_final_test',
        diagnosis_report_sha256=registration['diagnosis_report_sha256'],dataset_verified_manifest='runs/dataset_download/verified_manifest.json',
        dataset_manifest_sha256=sha(ROOT/'runs/dataset_download/verified_manifest.json'),
        augmentation_registration_sha256=sha(ROOT/'configs/augmentation_registration.json'),
        analysis_registration_sha256=sha(ROOT/'configs/analysis_registration.json'),
        resource_ladder_sha256=sha(ROOT/'evidence/resource_ladder.json'),
        norm_stats_sha256=sha(ROOT/'cache/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json'),
        source_hashes=read(ROOT/'third_party/openpi/robustness_patch_manifest.json'),
        checkpoint_rule='last step4999; full standard checkpoint persisted and checked before releasing RAM source',
        note='Single-seed controlled continuation from official pi05_libero; only registered extra augmentation differs.')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n');return plan_path


def make_space():
    data=ROOT/'cache/final_checkpoints';system=Path('/root/pi05_storage/checkpoints')
    data.mkdir(exist_ok=True,parents=True);system.mkdir(exist_ok=True,parents=True)
    if shutil.disk_usage(data).free+shutil.disk_usage(system).free<2*9386620499+6*1024**3:
        archive=Path('/root/pi05_storage/downloads/assets.zip')
        assert archive.resolve()==archive and archive.stat().st_size==6395849578
        assert read(ROOT/'runs/plus_assets_download/result.json')['status']=='complete_verified'
        assert read(ROOT/'runs/plus_assets_extract/result.json')['status']=='complete_verified'
        assert read(ROOT/'runs/plus_renderer_probe/result.json')['passed']
        checksum=sha(archive);assert checksum=='96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf'
        archive.unlink()
        (ROOT/'evidence/assets_archive_released.json').write_text(json.dumps({'path':str(archive),'bytes':6395849578,'sha256':checksum,'expanded_runtime_assets_retained':True,'reason':'Verified re-downloadable ZIP released to persist both full trained checkpoints.'},indent=2)+'\n')


def train_and_persist(model,state,plan_path):
    name=model+'_'+state['tag'];run=ROOT/'runs'/name;plan=read(plan_path)
    assert read(ROOT/'third_party/openpi/robustness_patch_manifest.json')==plan['source_hashes']
    write_state(state,'training_'+model)
    env=dict(os.environ,HF_LEROBOT_HOME=str(ROOT/'cache/training'))
    call([sys.executable,ROOT/'scripts/run_training.py','--openpi',ROOT/'third_party/openpi','--stage',model,
        '--batch',plan['batch_size'],'--steps',5000,'--max-minutes',600,'--experiment-name',name,
        '--training-plan',plan_path,'--dataset-manifest',ROOT/'runs/dataset_download/verified_manifest.json','--out',run],
        ROOT/'runs'/(name+'_supervisor.log'),37000,env)
    result=read(run/'result.json');assert result['success'];write_state(state,'persisting_'+model)
    source=Path(result['checkpoint_expected'])
    destination=(Path('/root/pi05_storage/checkpoints') if model=='B1' else ROOT/'cache/final_checkpoints')/name/'4999'
    spill=(ROOT/'cache/final_checkpoints' if model=='B1' else Path('/root/pi05_storage/checkpoints'))/(name+'_spill')
    manifest=run/'checkpoint_files.json'
    call([sys.executable,ROOT/'scripts/export_checkpoint.py',source,'--destination',destination,'--spill-directory',spill,'--out',manifest],run/'export.log',1800)
    call([PYTHON,ROOT/'scripts/verify_checkpoint_reload.py',destination,'--audit-dir',run,'--out',run/'reload_cpu.json'],run/'reload_cpu.log',1800,dict(os.environ,JAX_PLATFORMS='cpu'))
    action_env=dict(os.environ,OPENPI_AUG='default',XLA_PYTHON_CLIENT_MEM_FRACTION='.9');action_env.pop('JAX_PLATFORMS',None)
    call([PYTHON,ROOT/'scripts/checkpoint_action_sanity.py','--checkpoint',destination,'--observation',ROOT/'runs/policy_sanity_input/observation.npz','--out',run/'actions_sanity.json'],run/'actions_sanity.log',1800,action_env)
    assert read(run/'reload_cpu.json')['passed'] and read(run/'actions_sanity.json')['passed']
    receipt=read(manifest.with_suffix('.receipt.json'));assert receipt['status']=='complete_verified' and Path(receipt['source'])==source
    source.resolve().relative_to(Path('/dev/shm/pi05_checkpoints/pi05_libero_lora_ft').resolve())
    assert source.name=='4999' and source.parent.name==name
    shutil.rmtree(source)
    (run/'ram_source_released.json').write_text(json.dumps({'source':str(source),'persistent_checkpoint':str(destination),'export_manifest_sha256':sha(manifest),'cpu_reload_passed':True,'real_input_actions_passed':True},indent=2)+'\n')
    state['models'][model]={'run':str(run),'checkpoint':str(destination),'manifest':str(manifest)};write_state(state,model+'_trained_and_persisted')


def evaluate(model,kind,state,replan=5,denoise=10,serial=False):
    name=f'{model}_{"efficiency" if serial else kind}_R{replan}_N{denoise}_{state["tag"]}';out=ROOT/'runs'/name
    command=[sys.executable,ROOT/'scripts/run_evaluation.py','--kind',kind,'--suite','libero_spatial' if serial else 'all',
        '--model-id',model,'--evaluator-script',ROOT/'scripts/evaluate_v3.py','--replan-steps',replan,'--num-steps',denoise,
        '--max-hours',6,'--parallel-suites',1 if serial else 4,'--out',out]
    if kind=='plus':command+=['--plus-split','heldout']
    if model!='Ref':
        record=state['models'][model]
        for row in read(record['manifest']):
            path=Path(record['checkpoint'])/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
        command+=['--config','pi05_libero_lora_ft','--checkpoint',record['checkpoint'],'--checkpoint-manifest',record['manifest']]
    write_state(state,'evaluating_'+name);call(command,ROOT/'runs'/(name+'_supervisor.log'),22200)
    assert read(out/'result.json')['status']=='complete'
    state['evaluations'].append({'model':model,'kind':kind,'serial':serial,'replan_steps':replan,'num_steps':denoise,'out':str(out)})
    write_state(state,name+'_complete');return out


def check_ref_bridge(out):
    old=[json.loads(x) for x in (ROOT/'runs/ref_clean400_20261026T235719Z/clean400/episodes.jsonl').read_text().splitlines()]
    new=[json.loads(x) for p in (out/'episodes').glob('*/episodes.jsonl') for x in p.read_text().splitlines()]
    a={r['pair_id']:r for r in old};b={r['pair_id']:r for r in new};assert set(a)==set(b) and len(a)==400
    keys=['success','environment_steps','initial_state_sha256','settled_sim_state_sha256','initial_main_rgb_sha256','initial_wrist_rgb_sha256','prompt','flow_noise_identity']
    mismatches=[{'pair_id':key,'fields':[f for f in keys if a[key][f]!=b[key][f]]} for key in a if any(a[key][f]!=b[key][f] for f in keys)]
    proof={'passed':not mismatches,'paired_episodes':400,'fields':keys,'mismatches':mismatches,'scope':'Original serial versus instrumented concurrent evaluator; repeats never pooled as800 independent trials.'}
    (out/'baseline_equivalence.json').write_text(json.dumps(proof,indent=2)+'\n');assert not mismatches,'Ref bridge changed; investigate before heldout'


def main():
    assert not STATE.exists(),'Inspect existing pipeline state rather than create duplicate jobs'
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    state={'status':'running','tag':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),'models':{},'evaluations':[],'instance_left_on':True}
    write_state(state,'waiting_for_resource_selection')
    try:
        deadline=time.monotonic()+20000
        while True:
            resource=ROOT/'evidence/resource_ladder.json'
            if resource.exists() and read(resource)['status']=='complete':break
            if time.monotonic()>deadline:raise TimeoutError('Resource selection did not complete')
            time.sleep(10)
        plan_path=freeze_plan(read(resource));state['training_plan_sha256']=sha(plan_path);make_space()
        for model in ['B1','A1']:train_and_persist(model,state,plan_path)
        b1,a1=[Path(state['models'][m]['run']) for m in ['B1','A1']]
        assert read(b1/'params_initial.json')['parameters']==read(a1/'params_initial.json')['parameters']
        assert (b1/'first_batches.jsonl').read_bytes()==(a1/'first_batches.jsonl').read_bytes()
        state['matched_training_initialization_and_first10_inputs']=True
        for model in ['Ref','B1','A1']:
            clean=evaluate(model,'clean',state)
            if model=='Ref':check_ref_bridge(clean)
            evaluate(model,'plus',state)
        for replan,denoise in [(1,10),(5,10),(10,10),(5,3),(10,3)]:evaluate('A1','clean',state,replan,denoise,serial=True)
        state['status']='all_jobs_complete_pending_independent_acceptance';write_state(state,'analysis_ready')
    except Exception as exc:
        state.update(status='stopped_on_error',error=type(exc).__name__+': '+str(exc));write_state(state,'inspect_error');raise


if __name__=='__main__':main()
