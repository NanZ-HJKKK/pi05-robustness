"""Register camera-aware train-only augmentation after the default resource probe."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    upstream=ROOT/'third_party/openpi'
    manifest_path=upstream/'robustness_patch_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for name,expected in manifest.items():assert hashlib.sha256((upstream/name).read_bytes()).hexdigest()==expected,name
    model=upstream/'src/openpi/models/model.py';text=model.read_text()
    old='''            if "wrist" not in key:
                aug_rng = jax.random.fold_in(rng, 1000 + list(image_keys).index(key))
                image = augment_batch(aug_rng, image)'''
    new='''            aug_rng = jax.random.fold_in(rng, 1000 + list(image_keys).index(key))
            camera = "main" if key == "base_0_rgb" else ("wrist" if key == "left_wrist_0_rgb" else "unused")
            image = augment_batch(aug_rng, image, camera=camera)'''
    assert text.count(old)==1
    changed=text.replace(old,new);ast.parse(changed)
    backup=upstream/'robustness_camera_originals';assert not backup.exists();backup.mkdir()
    shutil.copy2(model,backup/'model.py')
    helper=upstream/'src/openpi/models/robust_augmentation.py';shutil.copy2(helper,backup/'robust_augmentation.py')
    model.write_text(changed);shutil.copy2(ROOT/'src/pi05_robustness/augmentation.py',helper)
    for name in ['src/openpi/models/model.py','src/openpi/models/robust_augmentation.py']:
        manifest[name]=hashlib.sha256((upstream/name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');print('Patched camera-aware training augmentation; default and inference bypass preserved.')


if __name__=='__main__':main()
