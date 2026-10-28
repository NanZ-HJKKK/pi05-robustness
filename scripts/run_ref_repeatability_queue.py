"""Bounded non-training diagnosis; preserve frozen model sources and checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'runs/ref_repeatability_diagnosis';assert not base.exists();base.mkdir()
    upstream=ROOT/'third_party/openpi';clone=ROOT/'cache/baseline_replay_package/openpi';assert not clone.exists()
    shutil.copytree(upstream/'src/openpi',clone,ignore=shutil.ignore_patterns('__pycache__'))
    original=ROOT/'runs/ref_clean400_20261026T235719Z/patched_upstream/src/openpi/models/model.py'
    shutil.copy2(original,clone/'models/model.py')
    variants=[('current08_a','current','.8'),('current08_b','current','.8'),('current09_a','current','.9'),('baseline09_a','baseline','.9')]
    records=[]
    for name,variant,fraction in variants:
        source=upstream/'src/openpi/models/model.py' if variant=='current' else clone/'models/model.py'
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        env=dict(os.environ,OPENPI_AUG='default',XLA_PYTHON_CLIENT_MEM_FRACTION=fraction,OPENPI_DATA_HOME=str(ROOT/'cache/openpi'),
            PYTHONPATH=os.pathsep.join(([str(clone.parent)] if variant=='baseline' else [])+[str(ROOT/'tools')]))
        with (base/(name+'.log')).open('w') as log:
            subprocess.run([str(upstream/'.venv/bin/python'),str(ROOT/'scripts/probe_ref_repeatability.py'),'--out',str(base/name),'--expected-model-sha',sha],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        records.append({'name':name,'result':json.loads((base/name/'result.json').read_text())})
    (base/'all_probes.json').write_text(json.dumps(records,indent=2)+'\n')
    (base/'status.json').write_text(json.dumps({'status':'probes_complete','next':'Compare arrays, identify numerical/source/concurrency effects before resuming heldout.'},indent=2)+'\n')


if __name__=='__main__':main()
