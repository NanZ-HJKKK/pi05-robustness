"""Verify cross-process/source actions using persisted GPU autotuning decisions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    prior=ROOT/'runs/ref_repeatability_diagnosis/all_probes.json';deadline=time.monotonic()+3000
    while not prior.exists():
        if time.monotonic()>deadline:raise TimeoutError('Prior probe did not finish')
        time.sleep(5)
    out=ROOT/'runs/persisted_autotune_probe';assert not out.exists();out.mkdir()
    cache=out/'ref_autotune.pb';upstream=ROOT/'third_party/openpi';clone=ROOT/'cache/baseline_replay_package/openpi'
    variants=[('capture','current','.8',True),('reload_a','current','.8',False),('reload_b','current','.8',False),('baseline_reload','baseline','.9',False)]
    records=[]
    for name,variant,fraction,capture in variants:
        source=(upstream/'src/openpi' if variant=='current' else clone)/'models/model.py'
        flags='--xla_gpu_dump_autotune_results_to='+str(cache) if capture else '--xla_gpu_load_autotune_results_from='+str(cache)+' --xla_gpu_require_complete_aot_autotune_results=true'
        env=dict(os.environ,OPENPI_AUG='default',XLA_PYTHON_CLIENT_MEM_FRACTION=fraction,OPENPI_DATA_HOME=str(ROOT/'cache/openpi'),XLA_FLAGS=flags,
            PYTHONPATH=os.pathsep.join(([str(clone.parent)] if variant=='baseline' else [])+[str(ROOT/'tools')]))
        command=[str(upstream/'.venv/bin/python'),str(ROOT/'scripts/probe_ref_repeatability.py'),'--out',str(out/name),'--expected-model-sha',hashlib.sha256(source.read_bytes()).hexdigest()]
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        record={'name':name,'exit_code':p.returncode,'xla_flags':flags}
        if p.returncode==0:record['result']=json.loads((out/name/'result.json').read_text())
        records.append(record);(out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        if p.returncode:raise RuntimeError('Autotune replay failed; inspect '+name+'.log')
        assert cache.is_file() and cache.stat().st_size>0
    import sys
    sys.path.insert(0,str(upstream/'.venv/lib/python3.11/site-packages'))
    # Hash comparison proves native action bytes, avoiding any tolerance-only acceptance.
    hashes=[r['result']['hashes'] for r in records];passed=all(h==hashes[0] for h in hashes[1:])
    report={'passed':passed,'records':records,'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'cache_bytes':cache.stat().st_size,
        'scope':'Three fresh processes, two source variants and two memory fractions; same real input and three PRNG seeds. No task outcomes used to select kernels.'}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    assert passed,'Persisted choices did not reproduce all action bytes'


if __name__=='__main__':main()
