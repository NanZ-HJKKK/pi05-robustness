"""Small parameter fingerprints for update/reload proof, independent of model score."""
import hashlib
import json
import math
from pathlib import Path


def memory_stats():
    import jax
    stats = jax.local_devices()[0].memory_stats() or {}
    return {key: int(value) for key, value in stats.items() if value is not None}


def capture_batch(batch, directory, step):
    if step >= 10:
        return
    import jax
    import numpy as np
    leaves = []
    for path, value in jax.tree_util.tree_flatten_with_path(batch)[0]:
        array = np.asarray(jax.device_get(value))
        leaves.append({"path": jax.tree_util.keystr(path), "shape": list(array.shape),
                       "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
    with (Path(directory) / "first_batches.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": step, "leaves": leaves}) + "\n")


def capture(state, config, directory, label):
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    import jax
    import jax.numpy as jnp
    import numpy as np
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    all_params = flatten_dict(state.params.filter(nnx.Param).to_pure_dict())
    trainable = flatten_dict(state.params.filter(config.trainable_filter).to_pure_dict())
    records = []
    for path, value in all_params.items():
        n = math.prod(value.shape)
        sample = np.asarray(jax.device_get(jnp.ravel(value)[:min(256, n)]))
        if not np.isfinite(sample.astype(np.float32)).all():
            raise FloatingPointError("Nonfinite sampled parameter at " + str(path))
        records.append({"path": list(path), "shape": list(value.shape), "dtype": str(sample.dtype),
                        "numel": n, "trainable": path in trainable, "sample_count": int(sample.size),
                        "sample_sha256": hashlib.sha256(sample.tobytes()).hexdigest()})
    result = {"label": label, "training_step": int(jax.device_get(state.step)),
              "total_parameters": sum(row["numel"] for row in records),
              "trainable_parameters": sum(row["numel"] for row in records if row["trainable"]),
              "sample_rule": "first256 flattened values of every parameter leaf; SHA256 over native dtype bytes",
              "memory_stats": memory_stats(), "parameters": records}
    (out / ("params_" + label + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    if label == "final":
        initial = json.loads((out / "params_initial.json").read_text())
        old = {tuple(row["path"]): row for row in initial["parameters"]}
        changed = []
        frozen_changed = []
        for row in records:
            previous = old[tuple(row["path"])]
            assert previous["shape"] == row["shape"] and previous["dtype"] == row["dtype"]
            if previous["sample_sha256"] != row["sample_sha256"]:
                (changed if row["trainable"] else frozen_changed).append(row["path"])
        proof = {"changed_sampled_trainable_leaves": len(changed), "changed_trainable_paths": changed,
                 "changed_sampled_frozen_leaves": len(frozen_changed), "changed_frozen_paths": frozen_changed,
                 "passed": bool(changed) and not frozen_changed,
                 "scope": "Every leaf sampled; not a full elementwise comparison of all parameters"}
        (out / "update_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
        if not proof["passed"]:
            raise AssertionError("Parameter update/freeze sample audit failed")
    return result
