"""Audit and package completed jobs; stop on any experiment error."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    deadline=time.monotonic()+100000
    state_path=ROOT/'evidence/full_experiment_state.json'
    while time.monotonic()<deadline:
        state=json.loads(state_path.read_text())
        if state['status']=='stopped_on_error':raise RuntimeError(state['error'])
        if state['status']=='all_jobs_complete_pending_independent_acceptance':break
        time.sleep(15)
    else:raise TimeoutError('Experiment completion window ended')
    out=ROOT/'runs/final_analysis';out.mkdir(exist_ok=True)
    with (out/'analysis.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_final.py'),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    with (out/'figures.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv_clean/bin/python'),str(ROOT/'scripts/plot_results.py'),str(out/'results.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300,env=dict(os.environ,MPLCONFIGDIR=str(ROOT/'cache/matplotlib')))
    roots=[Path(r['out']) for r in state['evaluations']]+[Path(r['run']) for r in state['models'].values()]+[out]
    roots += [Path(r['out']) for r in state.get('validation_runs',[])]
    files=set()
    for root in roots:
        for path in root.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:files.add(path)
    files.update(p for p in (ROOT/'configs').rglob('*') if p.is_file())
    files.update(p for p in (ROOT/'cache/inference_autotune').glob('*.pb') if p.is_file())
    for name in ['full_experiment_state.json','resource_ladder.json','exact_glass_activation.json','assets_archive_released.json','render_rounding_exception.json']:
        files.add(ROOT/'evidence'/name)
    archive=ROOT/'evidence'/('completed_study_'+state['tag']+'.zip');assert not archive.exists()
    manifest=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            relative=str(path.relative_to(ROOT));h=hashlib.sha256()
            with path.open('rb') as source,z.open(relative,'w') as destination:
                for block in iter(lambda:source.read(4*1024**2),b''):h.update(block);destination.write(block)
            manifest.append({'path':relative,'bytes':path.stat().st_size,'sha256':h.hexdigest()})
        z.writestr('payload_manifest.json',json.dumps(manifest,indent=2))
    h=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    receipt={'status':'ready_for_local_independent_acceptance','archive':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest(),'files':len(manifest),'instance_left_on':True}
    (ROOT/'evidence/completed_study_archive.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
