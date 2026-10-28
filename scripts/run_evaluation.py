"""One reproducible evaluation job with an owned policy server and bounded runtime."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from evaluate import plan_tasks
from run_clean_baseline import stop, interrupted

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["clean", "plus"], required=True)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--condition", choices=["normal", "black_main", "black_wrist"], default="normal")
    parser.add_argument("--model-id", choices=["Ref", "B1", "A1"], default="Ref")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "cache/checkpoints/pi05_libero")
    parser.add_argument("--checkpoint-manifest", type=Path, default=ROOT / "runs/setup_probe/checkpoint_verified.json")
    parser.add_argument("--num-steps", type=int, choices=[3, 10], default=10)
    parser.add_argument("--replan-steps", type=int, choices=[1, 5, 10], default=5)
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluator-script",type=Path,default=ROOT/"scripts/evaluate.py")
    parser.add_argument("--plus-split",choices=["diagnostic","heldout"],default="diagnostic")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--parallel-suites",type=int,choices=[1,4],default=1)
    args = parser.parse_args()
    if args.parallel_suites==4 and args.suite!="all":parser.error("Parallel suites requires --suite all")
    if args.max_hours <= 0 or (args.out.exists() and not args.resume):
        parser.error("Use a positive window and a fresh output directory")
    out = args.out.resolve()
    if args.resume and not (out/"result.json").exists():parser.error("Resume needs an existing run")
    tag=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.resume:
        previous=json.loads((out/"result.json").read_text())
        if previous["status"]=="complete":raise SystemExit("Run already complete")
        history=out/"resume_history";history.mkdir(exist_ok=True)
        shutil.copy2(out/"result.json",history/(tag+"_previous_result.json"))
    out.mkdir(parents=True,exist_ok=True)
    suffix="_resume_"+tag if args.resume else ""
    expected = len(plan_tasks(args.kind, args.suite, 1 if args.kind == "plus" else 10))
    if args.kind=="plus" and args.plus_split=="heldout":
        tasks=json.loads((ROOT/"configs/eval_subset.heldout.json").read_text())["tasks"]
        expected=sum(args.suite=="all" or task["suite"]==args.suite for task in tasks)
        if args.evaluator_script.name!="evaluate_v3.py":parser.error("Heldout requires registered evaluator v3")
    if expected == 0:
        raise SystemExit("Empty evaluation plan")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "episodes_planned": expected, "kind": args.kind, "condition": args.condition,
              "model_id": args.model_id, "training_run": False, "max_hours": args.max_hours,
              "evaluator_script_name":args.evaluator_script.name,"plus_split":args.plus_split,
              "resume":args.resume,
              "parallel_clients":args.parallel_suites,
              "timing_scope":"Concurrent suite evaluation; do not compare request/feedback latency to serial efficiency runs" if args.parallel_suites>1 else "One serial simulator client",
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result_path = out / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    server = client = None
    clients=[];client_logs=[]
    try:
        if shutil.disk_usage(ROOT).free < 2 * 1024**3:
            raise RuntimeError("Less than2GiB artifact space")
        if args.kind == "plus":
            proof = json.loads((ROOT / "runs/plus_assets_extract/result.json").read_text())
            if proof["status"] != "complete_verified":
                raise RuntimeError("Plus assets have not been verified/extracted")
        snapshot = (out/"resume_history"/tag/"source_snapshot") if args.resume else out/"source_snapshot"
        for directory in ["scripts", "src", "configs"]:
            for path in (ROOT / directory).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    dst = snapshot / path.relative_to(ROOT)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dst)
        for relative in ["src/openpi/models/model.py", "src/openpi/policies/policy.py", "src/openpi/training/config.py"]:
            dst = (out/"resume_history"/tag/"patched_upstream"/relative) if args.resume else out/"patched_upstream"/relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "third_party/openpi" / relative, dst)
        receipt=out/"checkpoint_verified.json"
        if receipt.exists():assert receipt.read_bytes()==args.checkpoint_manifest.read_bytes()
        else:shutil.copy2(args.checkpoint_manifest,receipt)
        environment_patch=ROOT/"evidence/exact_glass_activation.json"
        if args.kind=="plus" and environment_patch.exists():
            report["environment_patch_receipt_sha256"]=hashlib.sha256(environment_patch.read_bytes()).hexdigest()
            dst=snapshot.parent/"environment_patch.json";shutil.copy2(environment_patch,dst)
            for name in ["env_wrapper.py","exact_glass.py"]:
                shutil.copy2(ROOT/"third_party/libero_plus/libero/libero/envs"/name,snapshot.parent/name)
        openpi = ROOT / "third_party/openpi"
        server_env = dict(os.environ, OPENPI_AUG="default", XLA_PYTHON_CLIENT_MEM_FRACTION="0.8" if args.parallel_suites>1 else "0.9",
                          PYTHONPATH=str(ROOT / "tools"), OPENPI_DATA_HOME=str(ROOT / "cache/openpi"))
        if os.environ.get("OPENPI_FIXED_AUTOTUNE")=="1":
            registration_path=ROOT/"configs/inference_reproducibility_registration.json"
            registration=json.loads(registration_path.read_text())
            assert registration["status"]=="frozen" and not registration["heldout_outcomes_accessed"]
            cache=registration["caches"][args.config+"_N"+str(args.num_steps)]
            assert hashlib.sha256(Path(cache["path"]).read_bytes()).hexdigest()==cache["sha256"]
            server_env["XLA_FLAGS"]="--xla_gpu_load_autotune_results_from="+cache["path"]+" --xla_gpu_require_complete_aot_autotune_results=true"
            server_env["OPENPI_INFERENCE_AUTOTUNE_FILE"]=cache["path"]
            report["inference_reproducibility_registration_sha256"]=hashlib.sha256(registration_path.read_bytes()).hexdigest()
            report["inference_autotune_cache_sha256"]=cache["sha256"]
            shutil.copy2(registration_path,snapshot.parent/"inference_reproducibility_registration.json")
        metadata_path=out/("server_metadata"+suffix+".json")
        report["server_metadata_file"]=metadata_path.name
        with (out / ("server"+suffix+".log")).open("w") as log:
            server = subprocess.Popen([str(openpi / ".venv/bin/python"), str(ROOT / "scripts/serve.py"),
                "--model-id", args.model_id, "--config", args.config, "--num-steps", str(args.num_steps),
                "--port", "8001", "--checkpoint", str(args.checkpoint),
                "--checkpoint-manifest", str(args.checkpoint_manifest), "--metadata-out", str(metadata_path)],
                cwd=openpi, env=server_env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ready_deadline = min(deadline, time.monotonic() + 900)
        while time.monotonic() < ready_deadline:
            if server.poll() is not None:
                raise RuntimeError("Policy server failed before readiness")
            try:
                with http.open("http://127.0.0.1:8001/healthz", timeout=2) as response:
                    if response.status == 200 and response.read() == b"OK\n":
                        break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("Policy readiness window exceeded")
        config = out / "libero_config"
        config.mkdir(exist_ok=True)
        source = ROOT / "third_party/libero_plus" if args.kind == "plus" else openpi / "third_party/libero"
        package = source / "libero/libero"
        paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
                 "init_states": str(package / "init_files"), "assets": str(package / "assets"),
                 "datasets": str(ROOT / "cache/libero_datasets")}
        (config / "config.yaml").write_text(json.dumps(paths, indent=2) + "\n")
        client_env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", LIBERO_CONFIG_PATH=str(config),
                          PYTHONPATH=str(source), OMP_NUM_THREADS="4")
        python = ROOT / (".venv_plus/bin/python" if args.kind == "plus" else ".venv_clean/bin/python")
        remaining = deadline - time.monotonic() - 30
        if remaining < 60:
            raise TimeoutError("Evaluation window exhausted during startup")
        argv = [str(python), str(args.evaluator_script), "--kind", args.kind, "--suite", args.suite,
                "--model-id", args.model_id, "--condition", args.condition, "--port", "8001",
                "--trials", "1" if args.kind == "plus" else "10", "--num-steps", str(args.num_steps),
                "--replan-steps", str(args.replan_steps), "--max-minutes", str(remaining / 60), "--out", str(out / "episodes")]
        if args.evaluator_script.name=="evaluate_v3.py":argv.extend(["--plus-split",args.plus_split])
        suites=["libero_spatial","libero_object","libero_goal","libero_10"] if args.parallel_suites>1 else [args.suite]
        episode_dirs=[]
        for suite in suites:
            current=list(argv)
            directory=out/"episodes"/suite if args.parallel_suites>1 else out/"episodes"
            current[current.index("--suite")+1]=suite
            current[current.index("--out")+1]=str(directory)
            episode_dirs.append(directory)
            log=(out/("client_"+suite+suffix+".log")).open("w");client_logs.append(log)
            client=subprocess.Popen(current,cwd=ROOT,env=client_env,stdin=subprocess.DEVNULL,stdout=log,
                                    stderr=subprocess.STDOUT,start_new_session=True)
            clients.append(client)
        while any(proc.poll() is None for proc in clients):
            if time.monotonic() >= deadline:
                raise TimeoutError("Evaluation window ended")
            if server.poll() is not None:
                raise RuntimeError("Server exited during evaluation")
            if any(proc.poll() not in (None,0) for proc in clients):
                raise RuntimeError("A suite client failed; inspect raw records and its log")
            time.sleep(2)
        if any(proc.returncode for proc in clients):
            raise RuntimeError("Evaluation error; inspect client.log and raw episodes")
        rows = [json.loads(line) for directory in episode_dirs for line in (directory/"episodes.jsonl").read_text().splitlines()]
        attempts=len(rows)
        incomplete=[row for row in rows if row["status"]!="complete"]
        rows=[row for row in rows if row["status"]=="complete"]
        if len(rows) != expected:
            raise RuntimeError("Incomplete episode coverage")
        if len({row["pair_id"] for row in rows}) != expected:
            raise RuntimeError("Duplicate completed episode")
        report.update(status="complete", episodes_completed=expected, successes=sum(row["success"] for row in rows),
                      success_rate=sum(row["success"] for row in rows) / expected,
                      attempts_recorded=attempts,incomplete_attempts_retained=len(incomplete))
        subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), *map(str,episode_dirs),
                        "--out", str(out / "summary.json")], check=True, stdout=subprocess.DEVNULL, timeout=60)
    except Exception as exc:
        report.update(status="incomplete_or_failed", error=type(exc).__name__ + ": " + str(exc)[:200])
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for proc in clients:stop(proc)
        for log in client_logs:log.close()
        stop(server)
        report.update(wall_seconds=time.monotonic() - started, ended_at_utc=datetime.now(timezone.utc).isoformat(),
                      owned_processes_stopped=True, instance_left_on=True)
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
