"""Persist a complete RAM checkpoint, spilling files across owned disks if needed.

Every source/copy hash is checked; logical layout remains a standard checkpoint.
The original RAM checkpoint is never deleted by this script.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
ALLOWED=(ROOT/"cache/final_checkpoints",Path("/root/pi05_storage/checkpoints"))


def owned(path):
    path=path.resolve()
    if not any(path.is_relative_to(base.resolve()) for base in ALLOWED):
        raise ValueError("Checkpoint destination outside owned persistent roots")
    return path


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--destination",type=Path,required=True)
    p.add_argument("--spill-directory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--reserve-gib",type=float,default=3)
    args=p.parse_args();started=time.monotonic()
    source=args.source.resolve();source.relative_to(Path('/dev/shm/pi05_checkpoints').resolve())
    destination=owned(args.destination);spill=owned(args.spill_directory)
    if destination.exists() or spill.exists():raise SystemExit("Use fresh persistent destinations")
    destination.parent.mkdir(parents=True,exist_ok=True);spill.parent.mkdir(parents=True,exist_ok=True)
    if os.stat(destination.parent).st_dev==os.stat(spill.parent).st_dev:raise SystemExit("Spill must use a different filesystem")
    files=sorted(p for p in source.rglob('*') if p.is_file())
    assert files and (source/'params').is_dir() and (source/'train_state').is_dir()
    if any(p.is_symlink() for p in files):raise SystemExit("Inspect unexpected RAM checkpoint links")
    reserve=int(args.reserve_gib*1024**3)
    budgets=[shutil.disk_usage(destination.parent).free-reserve,shutil.disk_usage(spill.parent).free-reserve]
    total=sum(p.stat().st_size for p in files);needed=max(0,total-budgets[0]);spilled=set()
    for path in sorted(files,key=lambda x:x.stat().st_size,reverse=True):
        if needed<=0:break
        size=path.stat().st_size
        if size<=budgets[1]:
            spilled.add(path);budgets[1]-=size;needed-=size
    if needed>0:raise SystemExit("Insufficient persistent space with reserve; RAM source retained")
    destination.mkdir();spill.mkdir();records=[]
    for path in files:
        relative=path.relative_to(source);logical=destination/relative
        actual=(spill if path in spilled else destination)/relative
        actual.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(actual.parent).free-path.stat().st_size<reserve:raise RuntimeError("Space changed during export; source retained")
        temporary=actual.with_name(actual.name+'.copying');h=hashlib.sha256()
        with path.open('rb') as src,temporary.open('wb') as dst:
            for block in iter(lambda:src.read(8*1024**2),b''):
                h.update(block);dst.write(block)
            dst.flush();os.fsync(dst.fileno())
        assert temporary.stat().st_size==path.stat().st_size and digest(temporary)==h.hexdigest()
        temporary.replace(actual)
        if path in spilled:
            logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(actual)
        records.append({'path':str(relative),'bytes':path.stat().st_size,'sha256':h.hexdigest(),
                        'spill_target':str(actual) if path in spilled else None})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(records,indent=2)+'\n')
    executed_source=Path(__file__).read_bytes()
    args.out.with_suffix('.exporter_source.py').write_bytes(executed_source)
    receipt={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'complete_verified',
             'source':str(source),'destination':str(destination),'spill_directory':str(spill),
             'files':len(records),'bytes':total,'spilled_files':len(spilled),'reserve_bytes':reserve,
             'file_manifest_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
             'exporter_source_sha256':hashlib.sha256(executed_source).hexdigest(),
             'source_retained':True,'wall_seconds':time.monotonic()-started,
             'scope':'Full params, optimizer/train state and normalization assets; bit-identical file copies. Reload verification follows separately.'}
    args.out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
