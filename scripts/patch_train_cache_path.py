"""Redirect compilation cache only; preserve model/training numerical behavior."""
import ast
import hashlib
import json
from pathlib import Path
import sys


def main():
    repo=Path(sys.argv[1]).resolve();manifest_path=repo/"robustness_patch_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    for name,sha in manifest.items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha,name
    p=repo/"scripts/train.py";text=p.read_text()
    if 'os.environ.get("OPENPI_JAX_CACHE_DIR"' in text:
        print('Compilation cache path already configurable');return
    old='jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))'
    new='jax.config.update("jax_compilation_cache_dir", os.environ.get("OPENPI_JAX_CACHE_DIR", str(epath.Path("~/.cache/jax").expanduser())))'
    assert text.count(old)==1
    changed=text.replace(old,new,1);ast.parse(changed)
    backup=repo/"robustness_audit_originals/train_before_cache_path.py";assert not backup.exists();backup.write_text(text)
    p.write_text(changed);manifest['scripts/train.py']=hashlib.sha256(p.read_bytes()).hexdigest();manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Training compilation cache can use data disk; numerical code unchanged')


if __name__=='__main__':main()
