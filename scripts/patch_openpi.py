"""Apply only to verified pinned source; atomically validate all changes first."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

CONFIG = '''
    TrainConfig(
        name="pi05_libero_lora_ft",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
                                asset_id="physical-intelligence/libero"),
            base_config=DataConfig(prompt_from_task=True), extra_delta_transform=False,
        ),
        batch_size=16,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100, peak_lr=1e-5, decay_steps=5000, decay_lr=1e-6,
        ),
        num_train_steps=5000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None, seed=42, wandb_enabled=False,
    ),
'''


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Patch context does not match exactly: " + old[:80])
    return text.replace(old, new, 1)


def patched_sources(sources):
    if "robust_augmentation" in sources["src/openpi/models/model.py"]:
        raise ValueError("Augmentation patch already present")
    result = dict(sources)
    model = result["src/openpi/models/model.py"]
    model = once(model, "import abc\n", "import abc\nfrom openpi.models.robust_augmentation import augment_batch\n")
    model = once(model, "            # Back to [-1, 1].", '''            # Train-only, after default augmax; fold in a distinct stream per camera.
            if "wrist" not in key:
                aug_rng = jax.random.fold_in(rng, 1000 + list(image_keys).index(key))
                image = augment_batch(aug_rng, image)

            # Back to [-1, 1].''')
    result["src/openpi/models/model.py"] = model
    cfg = result["src/openpi/training/config.py"]
    cfg = once(cfg, '    TrainConfig(\n        name="pi05_libero",', CONFIG + '    TrainConfig(\n        name="pi05_libero",')
    result["src/openpi/training/config.py"] = cfg
    policy = result["src/openpi/policies/policy.py"]
    policy = once(policy, "        model_time = time.monotonic() - start_time", '''        # JAX dispatch is asynchronous: wait before stopping the sampling timer.
        if not self._is_pytorch_model:
            jax.block_until_ready(outputs["actions"])
        elif self._pytorch_device.startswith("cuda"):
            torch.cuda.synchronize(self._pytorch_device)
        model_time = time.monotonic() - start_time''')
    result["src/openpi/policies/policy.py"] = policy
    train = result["scripts/train.py"]
    train = once(train, "import functools\n", "import functools\nimport json\nimport os\nimport time\n")
    train = once(train, "    for step in pbar:\n        with sharding.set_mesh(mesh):", "    for step in pbar:\n        robust_step_start = time.perf_counter()\n        with sharding.set_mesh(mesh):")
    train = once(train, "        infos.append(info)", '''        if robust_metrics := os.environ.get("OPENPI_METRICS_JSONL"):
            jax.block_until_ready(info)
            robust_elapsed = time.perf_counter() - robust_step_start
            with open(robust_metrics, "a", encoding="utf-8") as robust_stream:
                robust_stream.write(json.dumps({"step": step, "step_seconds": robust_elapsed,
                    "metrics": {k: float(v) for k, v in jax.device_get(info).items()}}) + "\\n")
        infos.append(info)''')
    result["scripts/train.py"] = train
    for name, source in result.items():
        ast.parse(source, filename=name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("openpi", type=Path)
    args = parser.parse_args()
    target = args.openpi.resolve()
    lock = json.loads((ROOT / "evidence/upstream/manifest.json").read_text(encoding="utf-8"))["openpi"]
    names = ["src/openpi/models/model.py", "src/openpi/training/config.py", "src/openpi/policies/policy.py", "scripts/train.py"]
    sources = {}
    # Reject local edits and repeated patch application, before touching any file.
    for name in names:
        raw = (target / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != lock["files"][name]["sha256"]:
            raise SystemExit("Refusing unexpected/modified upstream: " + name)
        sources[name] = raw.decode("utf-8")
    patched = patched_sources(sources)
    backup = target / "robustness_originals"
    if backup.exists():
        raise SystemExit("Backup directory already exists")
    for name, text in patched.items():
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target / name, saved)
        (target / name).write_text(text, encoding="utf-8", newline="\n")
    shutil.copy2(ROOT / "src/pi05_robustness/augmentation.py", target / "src/openpi/models/robust_augmentation.py")
    record = {name: hashlib.sha256((target/name).read_bytes()).hexdigest() for name in patched}
    record["src/openpi/models/robust_augmentation.py"] = hashlib.sha256((target/"src/openpi/models/robust_augmentation.py").read_bytes()).hexdigest()
    (target / "robustness_patch_manifest.json").write_text(json.dumps(record,indent=2) + "\n", encoding="utf-8")
    print("Patched config, training-only augmentation and synchronized sampling latency.")


if __name__ == "__main__":
    main()
