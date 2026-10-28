"""Run authorized setup and 20 clean smoke episodes, with bounded child jobs."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def stop(proc):
    if proc is None or proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)


def main():
    tag=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=ROOT/"runs"/("phase_one_"+tag)
    out.mkdir(parents=True)
    report={"started_at_utc":datetime.now(timezone.utc).isoformat(),"status":"running",
            "authorized_budget_cny":30,"reported_gpu_hourly_price_cny":2.2,"episodes_planned":20,
            "episode_success_rate":None,"training_run":False,"model":"official pi05_libero"}
    result_path=out/"result.json"
    start=time.monotonic()
    server=client=None
    env=dict(os.environ)
    try:
        result_path.write_text(json.dumps(report,indent=2)+"\n")
        # Bound the authorized download/setup job; remaining smoke has a separate cap.
        setup=subprocess.run([sys.executable,str(ROOT/"scripts/setup_probe.py"),"--max-minutes","120"],env=env)
        if setup.returncode:raise RuntimeError("Setup incomplete; inspect runs/setup_probe")
        openpi=ROOT/"third_party/openpi"
        server_env=dict(env,OPENPI_AUG="default",XLA_PYTHON_CLIENT_MEM_FRACTION="0.9",PYTHONPATH=str(ROOT/"tools"),
                        OPENPI_DATA_HOME=str(ROOT/"cache/openpi"))
        metadata=out/"server_metadata.json"
        with (out/"server.log").open("w") as log:
            server=subprocess.Popen([str(openpi/".venv/bin/python"),str(ROOT/"scripts/serve.py"),
                "--model-id","Ref","--checkpoint",str(ROOT/"cache/checkpoints/pi05_libero"),
                "--metadata-out",str(metadata)],cwd=openpi,env=server_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            server_deadline=time.monotonic()+15*60
            while time.monotonic()<server_deadline:
                if server.poll() is not None:raise RuntimeError("Server exited before ready")
                try:
                    with socket.create_connection(("127.0.0.1",8000),timeout=2):break
                except OSError:time.sleep(2)
            else:raise TimeoutError("Server startup exceeded 15 minutes")
            config=out/"libero_config"
            config.mkdir()
            package=openpi/"third_party/libero/libero/libero"
            datasets=ROOT/"cache/libero_datasets"
            datasets.mkdir(parents=True,exist_ok=True)
            values={"benchmark_root":str(package),"bddl_files":str(package/"bddl_files"),
                    "init_states":str(package/"init_files"),"assets":str(package/"assets"),"datasets":str(datasets)}
            (config/"config.yaml").write_text(json.dumps(values,indent=2)+"\n")
            client_env=dict(env,MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",LIBERO_CONFIG_PATH=str(config),
                            PYTHONPATH=str(openpi/"third_party/libero"),OMP_NUM_THREADS="4")
            client_command=[str(ROOT/".venv_clean/bin/python"),str(ROOT/"scripts/evaluate.py"),
                "--kind","clean","--suite","libero_spatial","--model-id","Ref","--limit","20",
                "--max-minutes","30","--out",str(out/"smoke20")]
            with (out/"client.log").open("w") as client_log:
                client=subprocess.Popen(client_command,cwd=ROOT,env=client_env,stdout=client_log,stderr=subprocess.STDOUT,start_new_session=True)
                try:client.wait(timeout=31*60)
                except subprocess.TimeoutExpired:raise TimeoutError("Client exceeded smoke window")
            if client.returncode:raise RuntimeError("Client failed; inspect episode errors and client.log")
            episodes=[json.loads(line) for line in (out/"smoke20/episodes.jsonl").read_text().splitlines()]
            complete=[r for r in episodes if r["status"]=="complete"]
            if len(complete)!=20:raise RuntimeError("Smoke did not complete 20 episodes")
            report.update(status="smoke_complete",episodes_completed=20,successes=sum(r["success"] for r in complete),
                          episode_success_rate=sum(r["success"] for r in complete)/20,
                          note="First two Spatial tasks only; speed/pipeline smoke, not a four-suite baseline")
    except Exception as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        stop(client);stop(server)
        report["wall_seconds"]=time.monotonic()-start
        report["estimated_instance_charge_during_this_job_cny"]=report["wall_seconds"]/3600*2.2
        report["billing_note"]="Estimate for this process duration only, not the platform bill. Instance remains user-controlled."
        report["ended_at_utc"]=datetime.now(timezone.utc).isoformat()
        result_path.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if report["status"]=="smoke_complete" else 1)


if __name__=="__main__":main()
