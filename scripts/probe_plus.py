"""Render one diagnostic task per category twice, without using a policy."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    out = ROOT / "runs/plus_renderer_probe"
    out.mkdir(parents=True, exist_ok=True)
    assets = json.loads((ROOT / "runs/plus_assets_extract/result.json").read_text())
    assert assets["status"] == "complete_verified"
    source = ROOT / "third_party/libero_plus"
    package = source / "libero/libero"
    config = out / "libero_config"
    config.mkdir(exist_ok=True)
    paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
             "init_states": str(package / "init_files"), "assets": str(package / "assets"),
             "datasets": str(ROOT / "cache/libero_datasets")}
    (config / "config.yaml").write_text(json.dumps(paths) + "\n")
    os.environ.update(LIBERO_CONFIG_PATH=str(config), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="4")
    sys.path.insert(0, str(source))
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from evaluate import physical_bddl_path
    tasks = json.loads((ROOT / "configs/eval_subset.json").read_text())["tasks"]
    selected = {}
    for row in tasks:
        selected.setdefault(row["category"], row)
    records = []
    for index, (category, row) in enumerate(selected.items()):
        record = {"category": category, "task_name": row["task_name"], "suite": row["suite"], "passed": False}
        try:
            suite = benchmark.get_benchmark_dict()[row["suite"]](task_order_index=0)
            assert suite.get_task_names()[row["task_id"]] == row["task_name"]
            task = suite.get_task(row["task_id"])
            state = np.asarray(suite.get_task_init_states(row["task_id"])[0])
            bddl = package / "bddl_files" / task.problem_folder / task.bddl_file
            record["physical_bddl_sha256"] = hashlib.sha256(physical_bddl_path(bddl).read_bytes()).hexdigest()
            repeats = []
            for repeat in range(2):
                env = None
                try:
                    np.random.seed(7)
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                    env.seed(7)
                    env.reset()
                    obs = env.set_init_state(state)
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.] * 6 + [-1.])
                    current = {"state_sha256": hashlib.sha256(np.asarray(env.get_sim_state()).tobytes()).hexdigest()}
                    for key, camera in [("agentview_image", "main"), ("robot0_eye_in_hand_image", "wrist")]:
                        raw = np.ascontiguousarray(obs[key])
                        current[camera + "_raw_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
                        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(raw[::-1, ::-1]), 224, 224))
                        assert image.shape == (224, 224, 3) and np.isfinite(image).all() and image.std() > 0
                        Image.fromarray(image).save(out / ("%d_%d_%s.png" % (index, repeat, camera)))
                    repeats.append(current)
                finally:
                    if env is not None:
                        env.close()
            assert repeats[0] == repeats[1], "Repeated reset/state/image hashes differ"
            record.update(passed=True, repeats=repeats)
        except Exception:
            record["error"] = traceback.format_exc()
        records.append(record)
        print(json.dumps({"category": category, "passed": record["passed"], "error": record.get("error")}), flush=True)
    result = {"passed": len(records) == 7 and all(row["passed"] for row in records), "policy_used": False,
              "evaluation_episodes": 0, "scope": "Seven category render/reset probes, each repeated twice; no model success rate",
              "records": records, "wall_seconds": time.monotonic() - started}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
