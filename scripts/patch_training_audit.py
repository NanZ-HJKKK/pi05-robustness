"""Add synchronized optimizer/update and checkpoint fingerprints to pinned trainer."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    repo = Path(sys.argv[1]).resolve()
    manifest_path = repo / "robustness_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
    train = repo / "scripts/train.py"
    text = train.read_text()
    if "robust_audit.capture" in text:
        print("Training audit already installed and manifest verified")
        return
    original = text
    replacements = [
        ("import functools\n", "import functools\nimport math\nfrom openpi.training import robust_audit\n"),
        ('        "grad_norm": optax.global_norm(grads),', '        "grad_norm": optax.global_norm(grads),\n        "update_norm": optax.global_norm(updates),'),
        ('    start_step = int(train_state.step)', '    if audit_dir := os.environ.get("OPENPI_AUDIT_DIR"):\n        robust_audit.capture(train_state, config, audit_dir, "initial")\n\n    start_step = int(train_state.step)'),
        ('        robust_step_start = time.perf_counter()', '        if audit_dir := os.environ.get("OPENPI_AUDIT_DIR"):\n            robust_audit.capture_batch(batch, audit_dir, step)\n        robust_step_start = time.perf_counter()'),
        ('            jax.block_until_ready(info)', '            jax.block_until_ready((train_state, info))'),
        ('            robust_elapsed = time.perf_counter() - robust_step_start', '            robust_elapsed = time.perf_counter() - robust_step_start\n            robust_values = {k: float(v) for k, v in jax.device_get(info).items()}\n            if not all(math.isfinite(v) for v in robust_values.values()):\n                raise FloatingPointError("Nonfinite training metric")'),
        ('"metrics": {k: float(v) for k, v in jax.device_get(info).items()}', '"metrics": robust_values, "device_memory": robust_audit.memory_stats()'),
        ('    checkpoint_manager.wait_until_finished()', '    checkpoint_manager.wait_until_finished()\n    if audit_dir := os.environ.get("OPENPI_AUDIT_DIR"):\n        robust_audit.capture(train_state, config, audit_dir, "final")'),
    ]
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new, 1)
    ast.parse(text)
    backup = repo / "robustness_audit_originals/train.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    assert not backup.exists()
    backup.write_text(original)
    train.write_text(text)
    helper = repo / "src/openpi/training/robust_audit.py"
    shutil.copy2(ROOT / "src/pi05_robustness/training_audit.py", helper)
    for path in [train, helper]:
        manifest[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Trainer audit installed; model, optimizer, data and augmentation settings unchanged")


if __name__ == "__main__":
    main()
