"""Bounded Linux training supervisor. No downloads, overwrite, rent, or shutdown."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def interrupted(signum, frame):
    raise TimeoutError("Training supervisor interrupted by signal "+str(signum))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openpi",type=Path,required=True)
    p.add_argument("--stage",choices=["memtest","B1","A1"],required=True)
    p.add_argument("--batch",type=int,default=16)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--max-minutes",type=float,required=True)
    p.add_argument("--experiment-name",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--training-plan",type=Path)
    p.add_argument("--checkpoint-base-dir",type=Path,default=Path("/dev/shm/pi05_checkpoints"))
    p.add_argument("--dataset-manifest",type=Path,required=True)
    p.add_argument("--probe-augmentation",choices=["default","view_wrist_drop"],default="default")
    args=p.parse_args()
    if platform.system() != "Linux": raise SystemExit("Training is supported only on the Linux GPU host")
    if args.max_minutes<=0 or args.batch<1 or args.steps<1: p.error("Positive limits required")
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    mode=args.probe_augmentation if args.stage=="memtest" else "default"
    if args.stage!="memtest" and args.probe_augmentation!="default":p.error("Probe augmentation is only for resource tests")
    if args.stage != "memtest":
        if args.training_plan is None: p.error("Freeze a plan after diagnosis and memtest")
        plan=json.loads(args.training_plan.read_text(encoding="utf-8"))
        if plan["status"] != "frozen" or plan["batch_size"] != args.batch or plan["num_train_steps"] != args.steps:
            raise SystemExit("Frozen training plan mismatch")
        if not plan.get("dataset_verified_manifest") or not plan.get("diagnosis_report_sha256"):
            raise SystemExit("Data verification and diagnosis provenance missing")
        mode=plan["augmentation"] if args.stage == "A1" else "default"
        if mode not in ("default","view_drop","view_wrist_drop"): raise SystemExit("Register and implement another strategy first")
    elif args.steps != 100:
        raise SystemExit("Resource probe is fixed at 100 steps")
    if args.out.exists(): raise SystemExit("Choose a fresh output directory")
    checkpoint=args.checkpoint_base_dir/"pi05_libero_lora_ft"/args.experiment_name
    if checkpoint.exists(): raise SystemExit("Refusing to overwrite an experiment")
    patch_manifest=args.openpi/"robustness_patch_manifest.json"
    if not patch_manifest.exists(): raise SystemExit("Patch openpi first")
    patches=json.loads(patch_manifest.read_text(encoding="utf-8"))
    for file,sha in patches.items():
        if hashlib.sha256((args.openpi/file).read_bytes()).hexdigest()!=sha:
            raise SystemExit("Patched source drift: "+file)
    args.out.mkdir(parents=True)
    args.out=args.out.resolve()
    for folder in ["scripts","src","configs"]:
        for path in (ROOT/folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                dst=args.out/"source_snapshot"/path.relative_to(ROOT)
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    for name in patches:
        dst=args.out/"openpi_snapshot"/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.openpi/name,dst)
    shutil.copy2(patch_manifest,args.out/"openpi_snapshot/robustness_patch_manifest.json")
    env=dict(os.environ,OPENPI_AUG=mode,XLA_PYTHON_CLIENT_MEM_FRACTION="0.95",
        OPENPI_METRICS_JSONL=str(args.out/"steps.jsonl"),WANDB_MODE="disabled",HF_HUB_OFFLINE="1",
        OPENPI_DATA_HOME=str(ROOT/"cache/openpi"),OPENPI_AUDIT_DIR=str(args.out),HF_DATASETS_OFFLINE="1",
        HF_DATASETS_CACHE="/dev/shm/pi05_arrow",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4",
        OPENPI_JAX_CACHE_DIR=str(ROOT/"cache/jax_training"),CUDA_CACHE_PATH=str(ROOT/"cache/cuda"))
    # The parent must point HF_LEROBOT_HOME at the verified fixed-revision data.
    if not env.get("HF_LEROBOT_HOME"): raise SystemExit("Set HF_LEROBOT_HOME to the verified dataset root")
    data_manifest=json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    expected_revision=json.loads((ROOT/"evidence/resource_inventory.json").read_text(encoding="utf-8"))["dataset"]["revision"]
    if data_manifest["revision"] != expected_revision or data_manifest["repo"] != "physical-intelligence/libero":
        raise SystemExit("Unexpected dataset manifest/revision")
    data_root=Path(env["HF_LEROBOT_HOME"])/data_manifest["repo"]
    for entry in data_manifest["files"]:
        path=data_root/entry["path"]
        if path.stat().st_size!=entry["bytes"]:raise SystemExit("Dataset file changed: "+entry["path"])
    if args.stage != "memtest" and plan.get("dataset_manifest_sha256") != hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest():
        raise SystemExit("Frozen plan dataset hash mismatch")
    command=[str(args.openpi/".venv/bin/python"),str(ROOT/"scripts/train_local.py"),"pi05_libero_lora_ft",
        "--exp-name",args.experiment_name,"--batch-size",str(args.batch),"--num-train-steps",str(args.steps),
        "--checkpoint-base-dir",str(args.checkpoint_base_dir),"--keep-period","1000000000"]
    report={"stage":args.stage,"command":command,"augmentation":mode,"seed":42,"batch":args.batch,"steps":args.steps,
        "at_utc":datetime.now(timezone.utc).isoformat(),"max_minutes":args.max_minutes,"source_hashes":patches,
        "training_plan_sha256":hashlib.sha256(args.training_plan.read_bytes()).hexdigest() if args.training_plan else None,
        "status":"running","checkpoint_expected":str(checkpoint/(str(args.steps-1))),
        "dataset_manifest_sha256":hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest(),
        "checkpoint_storage":"shared memory during training; export/verify persistent inference checkpoint before releasing",
        "arrow_cache":"shared memory; raw Parquet and final inference checkpoints remain on disk"}
    (args.out/"run.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    start=time.monotonic()
    deadline=start+args.max_minutes*60
    with (args.out/"stdout.log").open("w",encoding="utf-8") as log:
        proc=subprocess.Popen(command,cwd=args.openpi,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        gpu=[]
        try:
            while proc.poll() is None and time.monotonic()<deadline:
                usage=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5)
                gpu.append({"seconds":time.monotonic()-start,"device_used_MiB":usage.stdout.strip()})
                time.sleep(min(2,max(0,deadline-time.monotonic())))
            if proc.poll() is None:
                report["status"]="timed_out"
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait(timeout=10)
            else:
                report["status"]="process_finished"
        except (TimeoutError,KeyboardInterrupt) as exc:
            report.update(status="interrupted",error=type(exc).__name__+": "+str(exc))
        finally:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            signal.signal(signal.SIGINT,signal.SIG_IGN)
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait(timeout=10)
    report.update(exit_code=proc.returncode,total_wall_seconds=time.monotonic()-start,gpu_samples=gpu)
    metrics=args.out/"steps.jsonl"
    rows=[json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()] if metrics.exists() else []
    report["recorded_training_steps"]=len(rows)
    report["median_step_seconds_excluding_first_5"]=statistics.median(r["step_seconds"] for r in rows[5:]) if len(rows)>5 else None
    report["checkpoint_exists"]=(checkpoint/str(args.steps-1)/"params").is_dir()
    proof_path=args.out/"update_proof.json"
    proof=json.loads(proof_path.read_text()) if proof_path.exists() else {}
    report["parameter_update_sample_audit_passed"]=proof.get("passed",False)
    report["continuous_step_indices"]=[r["step"] for r in rows]==list(range(args.steps))
    report["all_recorded_metrics_finite"]=all(math.isfinite(v) for row in rows for v in row["metrics"].values())
    report["peak_allocator_bytes_in_use"]=max((r.get("device_memory",{}).get("peak_bytes_in_use",0) for r in rows),default=0)
    report["success"]=proc.returncode==0 and len(rows)==args.steps and report["checkpoint_exists"] and proof.get("passed",False) and report["continuous_step_indices"] and report["all_recorded_metrics_finite"]
    (args.out/"result.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__=="__main__": main()
