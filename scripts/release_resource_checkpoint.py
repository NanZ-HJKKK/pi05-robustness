"""Release only a validated disposable resource-probe checkpoint from shared RAM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();run=args.run.resolve()
    result=json.loads((run/'result.json').read_text());assert result['stage']=='memtest' and result['success']
    for name in ['reload_cpu.json','actions_sanity.json']:
        assert json.loads((run/name).read_text())['passed'],name
    checkpoint=Path(result['checkpoint_expected']).resolve()
    base=Path('/dev/shm/pi05_checkpoints/pi05_libero_lora_ft').resolve()
    checkpoint.relative_to(base)
    assert checkpoint.parent.name==result['command'][result['command'].index('--exp-name')+1]
    assert checkpoint.name=='99' and not checkpoint.is_symlink()
    records=[]
    for path in sorted(checkpoint.rglob('*')):
        if path.is_file():
            assert not path.is_symlink()
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            records.append({'path':str(path.relative_to(checkpoint)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    (run/'resource_checkpoint_files.json').write_text(json.dumps(records,indent=2)+'\n')
    shutil.rmtree(checkpoint)
    receipt={'released':str(checkpoint),'bytes':sum(x['bytes'] for x in records),'files':len(records),
        'reason':'Disposable100step resource probe, not a selected or formal model. Parameter samples, losses, input hashes, full file hashes and successful CPU/action reload proofs retained.'}
    (run/'resource_checkpoint_released.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
