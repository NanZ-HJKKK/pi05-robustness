"""Closed-loop LIBERO evaluator; separate clean/Plus Python environments required."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def request_seed(pair_id, step, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{pair_id}:{step}".encode()).digest()[:4], "little")


def physical_bddl_path(path):
    # Plus camera/robot/noise variants are virtual filenames. Its constructor
    # rewrites them before opening the actual BDDL; hash that actual file.
    if "_view_" in path.name and "_initstate_" in path.name:
        return path.with_name(path.name.split("_view_")[0]+".bddl")
    return path


def plan_tasks(kind, suite, trials, plus_split="diagnostic"):
    prompts = json.loads((ROOT / "configs/base_prompts.json").read_text(encoding="utf-8"))
    if kind == "plus":
        filename = "eval_subset.heldout.json" if plus_split == "heldout" else "eval_subset.json"
        rows = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))["tasks"]
        return [dict(r, episode=0) for r in rows if suite == "all" or r["suite"] == suite]
    return [{"suite": s, "task_name": name, "base_task": name, "task_id": value["task_id"],
             "category": "clean", "episode": i, "init_state_index": i}
            for s, names in prompts.items() if suite == "all" or suite == s
            for name, value in names.items() for i in range(trials)]


def quat_to_axisangle(quat, np):
    q = np.asarray(quat).copy()
    q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3]**2)
    return np.zeros(3) if np.isclose(den,0) else q[:3] * 2*np.arccos(q[3]) / den


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["clean", "plus"], required=True)
    parser.add_argument("--suite", choices=["all", "libero_spatial", "libero_object", "libero_goal", "libero_10"], default="all")
    parser.add_argument("--model-id", choices=["Ref", "B1", "A1"], required=True)
    parser.add_argument("--plus-split", choices=["diagnostic", "heldout"], default="diagnostic")
    parser.add_argument("--condition", choices=["normal", "black_main", "black_wrist"], default="normal")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, choices=[1,5,10], default=5)
    parser.add_argument("--num-steps", type=int, choices=[3,10], default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-minutes", type=float, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_minutes <= 0 or args.trials < 1 or (args.limit is not None and args.limit < 1):
        parser.error("limits must be positive")
    if args.kind == "plus" and args.condition != "normal":
        parser.error("Run camera diagnostics on clean Spatial")
    if args.kind == "plus" and args.trials != 1:
        parser.error("Plus protocol requires --trials 1")
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import libero
    config = json.loads((ROOT/"configs/protocol.json").read_text(encoding="utf-8"))
    for filename, key in [("eval_subset.json", "subset_sha256"), ("base_prompts.json", "prompts_sha256")]:
        if digest((ROOT/"configs"/filename).read_bytes()) != config[key]:
            raise SystemExit("Frozen protocol mismatch: " + filename)
    if args.kind == "plus":
        registration = json.loads((ROOT/"configs/holdout_registration.json").read_text(encoding="utf-8"))
        assert digest((ROOT/"configs/protocol.json").read_bytes()) == registration["base_protocol_sha256"]
        assert digest((ROOT/"configs/eval_subset.heldout.json").read_bytes()) == registration["heldout_subset_sha256"]
        assert digest((ROOT/"configs/eval_subset.json").read_bytes()) == registration["diagnostic_subset_sha256"]
    rows = plan_tasks(args.kind, args.suite, args.trials, args.plus_split)
    if args.limit:
        rows = rows[:args.limit]
    # Upstream client retries forever and recv() has no timeout. Bound both to
    # the authorized window instead of silently leaving a paid job waiting.
    deadline = time.monotonic() + args.max_minutes*60
    class BoundedClient(WebsocketClientPolicy):
        def _wait_for_server(self):
            import websockets.sync.client
            from openpi_client import msgpack_numpy
            conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None,
                                                  open_timeout=min(20,args.max_minutes*60))
            metadata = msgpack_numpy.unpackb(conn.recv(timeout=max(.1,min(20,deadline-time.monotonic()))))
            return conn, metadata

        def infer(self, obs):
            from openpi_client import msgpack_numpy
            self._ws.send(self._packer.pack(obs))
            # The first request compiles the full JAX policy. It may need
            # several minutes, while every request still obeys the job window.
            cap = 600 if getattr(self,"_sent_requests",0)==0 else 120
            self._sent_requests = getattr(self,"_sent_requests",0)+1
            message = self._ws.recv(timeout=max(.1,min(cap,deadline-time.monotonic())))
            if isinstance(message,str):
                raise RuntimeError(message)
            return msgpack_numpy.unpackb(message)

    client = BoundedClient(args.host, args.port)
    meta = client.get_server_metadata()
    if meta.get("model_id") != args.model_id or meta.get("num_steps") != args.num_steps or not meta.get("latency_synchronized"):
        raise SystemExit("Server model/num_steps/timing metadata mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"videos").mkdir(exist_ok=True)
    settings = vars(args).copy()
    settings["out"] = str(args.out.resolve())
    settings.pop("max_minutes")  # May extend an explicitly authorized resume window.
    provenance = {"settings": settings, "server": meta, "expected_episodes": len(rows),
        "libero_module": str(Path(benchmark.__file__).resolve()),
        "protocol_sha256": digest((ROOT/"configs/protocol.json").read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()), "evaluator_filename": Path(__file__).name,
        "python": sys.version, "render_backend": os.environ.get("MUJOCO_GL")}
    old = args.out/"run.json"
    if old.exists():
        previous = json.loads(old.read_text(encoding="utf-8"))
        for key in ["settings", "protocol_sha256", "script_sha256", "libero_module"]:
            if provenance[key] != previous[key]:
                raise SystemExit("Unsafe resume: changed " + key)
        # Server restarts are allowed, but never changing model/checkpoint/hardware.
        for key in ["model_id", "checkpoint", "checkpoint_receipt_sha256", "config", "num_steps", "gpu", "source_sha256", "server_script_sha256"]:
            if meta[key] != previous["server"][key]:
                raise SystemExit("Unsafe resume: server changed " + key)
    else:
        old.write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        (args.out/"requirements.freeze.txt").write_text("\n".join(packages)+"\n",encoding="utf-8")
    output = args.out/"episodes.jsonl"
    finished = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] == "complete":
                if row["pair_id"] in finished:
                    raise SystemExit("Duplicate completed episode")
                finished.add(row["pair_id"])
    prompts = json.loads((ROOT/"configs/base_prompts.json").read_text(encoding="utf-8"))
    suites = {}
    for task_row in rows:
        suite_name = task_row["suite"]
        pair_id = f"{args.kind}/{suite_name}/{task_row['task_name']}/{task_row['init_state_index']}"
        if pair_id in finished:
            continue
        if time.monotonic() >= deadline:
            print("Time window ended; resume with identical settings", flush=True)
            return
        if suite_name not in suites:
            suites[suite_name] = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        suite = suites[suite_name]
        # Name is authoritative; Plus classification IDs are not runtime indices.
        names = suite.get_task_names()
        if task_row["task_name"] not in names:
            raise SystemExit("Wrong clean/Plus environment or task version")
        task_id = names.index(task_row["task_name"])
        if task_id != task_row["task_id"]:
            raise SystemExit("Task ordering differs from locked upstream")
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        state_index = task_row["init_state_index"]
        initial_state = np.asarray(states[state_index])
        prompt = str(task.language) if task_row["category"] == "Language Instructions" else prompts[suite_name][task_row["base_task"]]["policy_prompt"]
        if args.kind == "clean" and prompt != str(task.language):
            raise SystemExit("Official clean prompt differs from locked mapping")
        bddl = Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
        record = dict(task_row, pair_id=pair_id, model_id=args.model_id, condition=args.condition,
            replan_steps=args.replan_steps, num_steps=args.num_steps, seed=args.seed,
            prompt=prompt, original_task_language=str(task.language),
            initial_state_sha256=digest(initial_state.tobytes()), bddl_sha256=digest(physical_bddl_path(bddl).read_bytes()),
            requested_bddl_name=task.bddl_file,
            server_run_id=meta["server_run_id"], started_at_utc=datetime.now(timezone.utc).isoformat(),
            success=None, status="running", inference=[], environment_steps=0)
        noise_identity = f"{suite_name}/{task_row['base_task']}/{state_index}"
        record["flow_noise_identity"] = noise_identity
        env, frames, success = None, [], False
        started = time.perf_counter()
        try:
            np.random.seed(args.seed)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(args.seed)
            env.reset()
            obs = env.set_init_state(initial_state)
            for _ in range(10):
                obs, _, _, _ = env.step([0.]*6+[-1.])
            record["settled_sim_state_sha256"] = digest(np.asarray(env.get_sim_state()).tobytes())
            record["initial_main_rgb_sha256"] = digest(np.ascontiguousarray(obs["agentview_image"]).tobytes())
            record["initial_wrist_rgb_sha256"] = digest(np.ascontiguousarray(obs["robot0_eye_in_hand_image"]).tobytes())
            plan = deque()
            video_id = digest(pair_id.encode())[:20]
            video_path = args.out/"videos"/(video_id+".mp4")
            writer = imageio.get_writer(str(video_path), fps=10)
            control_started = time.perf_counter()
            record["actions"] = []
            record["rewards"] = []
            record["done_signals"] = []
            try:
                for step in range(config["inference"]["max_steps"][suite_name]):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Explicit evaluation time window ended")
                    main_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1,::-1]),224,224))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1,::-1]),224,224))
                    if args.condition == "black_main":
                        main_img = np.zeros_like(main_img)
                    if args.condition == "black_wrist":
                        wrist_img = np.zeros_like(wrist_img)
                    writer.append_data(np.concatenate([main_img,wrist_img],axis=1))
                    if not plan:
                        element = {"observation/image": main_img, "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate([obs["robot0_eef_pos"],quat_to_axisangle(obs["robot0_eef_quat"],np),obs["robot0_gripper_qpos"]]),
                            "prompt": prompt, "__request_seed": request_seed(noise_identity, step, args.seed)}
                        if not np.isfinite(element["observation/state"]).all():
                            raise ValueError("Nonfinite simulator proprioception")
                        infer_start = time.perf_counter()
                        response = client.infer(element)
                        request_ms = (time.perf_counter()-infer_start)*1000
                        actions = np.asarray(response["actions"])
                        if actions.ndim != 2 or actions.shape[0] < args.replan_steps or actions.shape[1] != 7 or not np.isfinite(actions).all():
                            raise ValueError("Invalid action chunk")
                        timing = dict(response["timing"], request_wall_ms=request_ms, environment_step=step,
                            request_start_elapsed_seconds=infer_start-control_started,
                            request_end_elapsed_seconds=time.perf_counter()-control_started,
                            request_seed_uint32=element["__request_seed"],
                            observed_proprio=element["observation/state"].tolist())
                        if not all(np.isfinite(timing[k]) and timing[k] > 0 for k in ["sampling_ms","policy_wall_ms","request_wall_ms"]):
                            raise ValueError("Invalid timing")
                        record["inference"].append(timing)
                        plan.extend(actions[:args.replan_steps])
                    action = plan.popleft().tolist()
                    obs, reward, done, _ = env.step(action)
                    if not np.isfinite(reward):
                        raise ValueError("Nonfinite simulator reward")
                    record["actions"].append(action)
                    record["rewards"].append(float(reward))
                    record["done_signals"].append(bool(done))
                    record["environment_steps"] += 1
                    if done:
                        success = True
                        break
            finally:
                record["control_loop_wall_seconds"] = time.perf_counter()-control_started
                writer.close()
            record.update(status="complete", success=success, video=str(video_path.resolve()))
        except Exception as exc:
            record.update(status="interrupted" if isinstance(exc, TimeoutError) else "error",
                          error=f"{type(exc).__name__}: {exc}")
        finally:
            if env is not None:
                try: env.close()
                except Exception as exc:
                    record["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                    record["observed_success_before_cleanup"] = record.get("success")
                    record.update(status="error",success=None,error=record["cleanup_error"])
            record["episode_wall_seconds"] = time.perf_counter()-started
            append(output, record)
        print(json.dumps({k:record.get(k) for k in ["pair_id","status","success","environment_steps","error"]}),flush=True)
        if record["status"] != "complete":
            raise SystemExit("Stopped on incomplete episode; inspect log before resume")
    print("All scheduled episodes complete", flush=True)


if __name__ == "__main__":
    main()
