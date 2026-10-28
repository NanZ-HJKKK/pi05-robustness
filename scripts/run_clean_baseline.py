"""Run the authorized 400-episode Ref baseline within a three-hour window."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PORT = 8001
MAX_SECONDS = 10740  # Leave one minute for termination and receipts within 3h.


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def interrupted(signum, frame):
    raise TimeoutError("Supervisor interrupted by signal " + str(signum))


def main():
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "runs" / ("ref_clean400_" + tag)
    out.mkdir(parents=True)
    (ROOT / "evidence/ref_clean400_current_run.txt").write_text(str(out) + "\n")
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "episodes_planned": 400, "model": "official pi05_libero", "training_run": False,
              "scope": "Four clean LIBERO suites, ten tasks each, ten initial states per task",
              "authorized_gpu_window_seconds": 10800, "authorized_compute_cap_cny": 6.6,
              "reported_hourly_price_cny": 2.2, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result_path = out / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    server = client = None
    try:
        if shutil.disk_usage(ROOT).free < 3 * 1024**3:
            raise RuntimeError("Less than 3GiB free for baseline artifacts")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", PORT))
        receipt = ROOT / "runs/setup_probe/checkpoint_verified.json"
        for item in json.loads(receipt.read_text()):
            path = ROOT / "cache" / item["name"]
            # Cache layout stores checkpoints directly under cache/checkpoints.
            if path.stat().st_size != item["size"]:
                raise RuntimeError("Checkpoint size changed: " + str(path))
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    h.update(chunk)
            if h.hexdigest() != item["sha256"]:
                raise RuntimeError("Checkpoint hash changed: " + str(path))
            if time.monotonic() >= deadline:
                raise TimeoutError("Time window ended during preflight")
        snapshot = out / "source_snapshot"
        for folder in ["scripts", "src", "configs"]:
            for path in (ROOT / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    target = snapshot / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        shutil.copy2(receipt, out / "checkpoint_verified.json")
        openpi = ROOT / "third_party/openpi"
        server_env = dict(os.environ, OPENPI_AUG="default", XLA_PYTHON_CLIENT_MEM_FRACTION="0.9",
                          PYTHONPATH=str(ROOT / "tools"), OPENPI_DATA_HOME=str(ROOT / "cache/openpi"))
        metadata = out / "server_metadata.json"
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen([str(openpi / ".venv/bin/python"), str(ROOT / "scripts/serve.py"),
                "--model-id", "Ref", "--port", str(PORT), "--checkpoint", str(ROOT / "cache/checkpoints/pi05_libero"),
                "--metadata-out", str(metadata)], cwd=openpi, env=server_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ready_deadline = min(deadline, time.monotonic() + 900)
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < ready_deadline:
            if server.poll() is not None:
                raise RuntimeError("Policy server exited before readiness")
            try:
                with http.open("http://127.0.0.1:%d/healthz" % PORT, timeout=2) as response:
                    if response.status == 200 and response.read() == b"OK\n":
                        break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("Policy server startup exceeded its window")
        config = out / "libero_config"
        config.mkdir()
        package = openpi / "third_party/libero/libero/libero"
        values = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
                  "init_states": str(package / "init_files"), "assets": str(package / "assets"),
                  "datasets": str(ROOT / "cache/libero_datasets")}
        (config / "config.yaml").write_text(json.dumps(values, indent=2) + "\n")
        client_env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", LIBERO_CONFIG_PATH=str(config),
                          PYTHONPATH=str(openpi / "third_party/libero"), OMP_NUM_THREADS="4")
        remaining = deadline - time.monotonic()
        if remaining < 60:
            raise TimeoutError("Insufficient time left to start evaluation")
        with (out / "client.log").open("w") as log:
            client = subprocess.Popen([str(ROOT / ".venv_clean/bin/python"), str(ROOT / "scripts/evaluate.py"),
                "--kind", "clean", "--suite", "all", "--model-id", "Ref", "--port", str(PORT),
                "--max-minutes", str(max(0.1, (remaining - 15) / 60)), "--out", str(out / "clean400")],
                cwd=ROOT, env=client_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while client.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Authorized baseline window ended")
            if server.poll() is not None:
                raise RuntimeError("Policy server exited during evaluation")
            time.sleep(2)
        if client.returncode:
            raise RuntimeError("Evaluation stopped; inspect client.log and episode rows")
        rows = [json.loads(line) for line in (out / "clean400/episodes.jsonl").read_text().splitlines()]
        if len(rows) != 400 or any(row["status"] != "complete" for row in rows):
            raise RuntimeError("Expected 400 completed episodes")
        if len({row["pair_id"] for row in rows}) != 400:
            raise RuntimeError("Duplicate formal episode")
        report.update(status="clean400_complete", episodes_completed=400, successes=sum(r["success"] for r in rows),
                      episode_success_rate=sum(r["success"] for r in rows) / 400)
    except Exception as exc:
        report.update(status="incomplete_or_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Ensure a second termination signal cannot interrupt owned-child cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop(client)
        stop(server)
        report.update(wall_seconds=time.monotonic() - started, ended_at_utc=datetime.now(timezone.utc).isoformat(),
                      owned_processes_stopped=True, instance_left_on=True)
        report["estimated_compute_during_job_cny"] = report["wall_seconds"] / 3600 * 2.2
        report["billing_note"] = "Job-duration estimate only; excludes other instance time and storage."
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "clean400_complete" else 1)


if __name__ == "__main__":
    main()
