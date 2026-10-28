"""Authorized phase-one Linux setup. Deadline and free-space guard; persistent logs.

Installs isolated server/client environments and downloads only official source
and pi05_libero checkpoint. Does not download training data or Plus assets.
"""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-minutes",type=float,default=90)
    args=parser.parse_args()
    if platform.system()!="Linux":raise SystemExit("Linux only")
    deadline=time.monotonic()+args.max_minutes*60
    run=ROOT/"runs/setup_probe"
    run.mkdir(parents=True,exist_ok=True)
    result={"status":"running","started_at_utc":datetime.now(timezone.utc).isoformat(),
            "max_minutes":args.max_minutes,"training_dataset_downloaded":False,"steps":[]}
    lock=json.loads((ROOT/"evidence/upstream/manifest.json").read_text())
    env=dict(os.environ,GIT_LFS_SKIP_SMUDGE="1",UV_CACHE_DIR=str(ROOT/"cache/uv"),
        PIP_CACHE_DIR=str(ROOT/"cache/pip"),UV_HTTP_TIMEOUT="60",UV_CONCURRENT_DOWNLOADS="4",
        PYTHONPATH=str(ROOT/"tools"),OPENPI_DATA_HOME=str(ROOT/"cache/openpi"))
    # The platform's international proxy rejects this domestic package mirror.
    env["no_proxy"]=env.get("no_proxy","")+",mirrors.aliyun.com"

    def check():
        if time.monotonic()>deadline:raise TimeoutError("Setup deadline reached")
        if shutil.disk_usage(ROOT).free<2*2**30:raise RuntimeError("Stopping before data disk has less than 2GiB free")

    def save():
        (run/"result.json").write_text(json.dumps(result,indent=2)+"\n")

    def command(label,argv,cwd=ROOT):
        check()
        entry={"name":label,"command":argv,"status":"running","start_utc":datetime.now(timezone.utc).isoformat()}
        result["steps"].append(entry);save()
        with (run/(label+".log")).open("w") as output:
            process=subprocess.Popen(argv,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while process.poll() is None:
                    check();time.sleep(2)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        entry.update(status="complete" if process.returncode==0 else "failed",exit_code=process.returncode)
        save()
        if process.returncode:raise RuntimeError(label+" failed; see log")

    def unpack(source,destination):
        if (destination/".snapshot_complete").exists():return
        destination.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                relative=Path(*Path(info.filename).parts[1:])
                path=destination/relative
                if not path.resolve().is_relative_to(destination.resolve()):raise ValueError("Unsafe archive member")
                if info.is_dir():path.mkdir(parents=True,exist_ok=True)
                else:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(info) as src,path.open("wb") as dst:shutil.copyfileobj(src,dst)
        (destination/".snapshot_complete").write_text("verified archive extracted\n")

    def download(url,path,expected_md5=None,expected_size=None,cap=None):
        path.parent.mkdir(parents=True,exist_ok=True)
        def valid():
            if not path.exists():return False
            if expected_size is not None and path.stat().st_size!=expected_size:return False
            if expected_md5:
                h=hashlib.md5()
                with path.open("rb") as f:
                    for block in iter(lambda:f.read(8*1024*1024),b""):check();h.update(block)
                return base64.b64encode(h.digest()).decode()==expected_md5
            return True
        if valid():return
        partial=path.with_suffix(path.suffix+".part")
        with urllib.request.urlopen(url,timeout=60) as response,partial.open("wb") as f:
            size=0;h=hashlib.md5()
            while True:
                check();block=response.read(4*1024*1024)
                if not block:break
                size+=len(block)
                if cap and size>cap:raise ValueError("Download exceeds expected source size")
                f.write(block);h.update(block)
        if expected_size is not None and size!=expected_size:raise ValueError("Download size mismatch")
        if expected_md5 and base64.b64encode(h.digest()).decode()!=expected_md5:raise ValueError("Download checksum mismatch")
        partial.replace(path)

    try:
        openpi=ROOT/"third_party/openpi"
        archive_meta=json.loads((ROOT/"evidence/openpi_source.archive.json").read_text())
        archive=ROOT/"evidence/openpi_source.zip"
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==archive_meta["sha256"]
        unpack(archive,openpi)
        if not (openpi/"robustness_patch_manifest.json").exists():
            command("patch",[sys.executable,str(ROOT/"scripts/patch_openpi.py"),str(openpi)])
        uv=[sys.executable,"-m","uv"]
        if not (ROOT/"tools/uv").exists():
            wheel=ROOT/"evidence/uv-0.9.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
            if wheel.exists():
                expected="d89f88df09d571f6d06228b32a6a71100905eb64343247317d363bcd774ee870"
                if hashlib.sha256(wheel.read_bytes()).hexdigest()!=expected:raise ValueError("uv wheel hash mismatch")
                command("uv_install",[sys.executable,"-m","pip","install","--no-index","--target",str(ROOT/"tools"),str(wheel)])
            else:
                command("uv_install",[sys.executable,"-m","pip","install","--index-url","https://mirrors.aliyun.com/pypi/simple","--target",str(ROOT/"tools"),"--require-hashes","-r",str(ROOT/"configs/uv_bootstrap.requirements.txt")])
        # Only transport URLs change. Versions, dependency graph and official
        # distribution hashes remain identical; uv still verifies every hash.
        dependency_lock=openpi/"uv.lock"
        original=dependency_lock.read_bytes()
        mirrored=original.replace(b"https://files.pythonhosted.org/",b"https://mirrors.aliyun.com/pypi/")
        if mirrored!=original:
            (openpi/"uv.lock.upstream").write_bytes(original)
            dependency_lock.write_bytes(mirrored)
            result["dependency_transport"]={"original_lock_sha256":hashlib.sha256(original).hexdigest(),
                "mirror_lock_sha256":hashlib.sha256(mirrored).hexdigest(),"versions_and_hashes_changed":False,
                "url_replacements":original.count(b"https://files.pythonhosted.org/")}
            save()
        command("server_env",uv+["sync","--frozen","--no-dev","--python","3.11"],openpi)
        command("server_check",uv+["pip","check","--python",str(openpi/".venv/bin/python")])
        command("server_inventory",[str(openpi/".venv/bin/python"),str(ROOT/"scripts/preflight.py"),"--out",str(run/"server_inventory.json")])
        # Official clean LIBERO source, independent from the user's old checkout.
        libero_archive=ROOT/"cache/libero_source.zip"
        download(f"https://codeload.github.com/{lock['libero']['repo']}/zip/{lock['libero']['commit']}",libero_archive,cap=600*1024**2)
        unpack(libero_archive,openpi/"third_party/libero")
        clean=ROOT/".venv_clean"
        if not clean.exists():command("client_venv",uv+["venv","--python","3.8",str(clean)])
        # This environment runs MuJoCo and loads CPU initial states only. Policy
        # inference runs on the separate JAX GPU server. Keep torch 1.11 APIs,
        # using its official CPU build to avoid 1.5GiB of unused CUDA binaries.
        client_requirements=ROOT/"configs/libero_client_cpu.requirements.txt"
        client_text=(openpi/"examples/libero/requirements.txt").read_text()
        client_text=client_text.replace("+cu113","+cpu")
        client_requirements.write_text(client_text)
        result["client_adaptation"]="Official Python3.8 pins; Torch/TorchVision/TorchAudio use same-version CPU builds; EGL still renders on GPU; pi05 policy remains JAX CUDA."
        save()
        command("client_env",uv+["pip","install","--python",str(clean/"bin/python"),
            "-r",str(client_requirements),"-r",str(openpi/"third_party/libero/requirements.txt"),
            "--index-url","https://mirrors.aliyun.com/pypi/simple",
            "--torch-backend","cpu"])
        command("client_packages",uv+["pip","install","--python",str(clean/"bin/python"),"-e",str(openpi/"packages/openpi-client"),"-e",str(openpi/"third_party/libero")])
        command("client_check",uv+["pip","check","--python",str(clean/"bin/python")])
        inventory=json.loads((ROOT/"evidence/resource_inventory.json").read_text())
        if not (ROOT/"tools/google_crc32c").exists():
            command("checksum_library",uv+["pip","install","--python",sys.executable,"--target",str(ROOT/"tools"),
                "--index-url","https://mirrors.aliyun.com/pypi/simple","google-crc32c==1.7.1"])
        from fetch_checkpoint import download_checkpoint
        verified=download_checkpoint(inventory["checkpoint"]["files"],ROOT/"cache",check)
        (run/"checkpoint_verified.json").write_text(json.dumps(verified,indent=2)+"\n")
        alias=ROOT/"cache/openpi/openpi-assets/checkpoints/pi05_libero"
        alias.parent.mkdir(parents=True,exist_ok=True)
        if not alias.exists():alias.symlink_to(ROOT/"cache/checkpoints/pi05_libero",target_is_directory=True)
        result["status"]="setup_complete_checkpoint_verified"
    except Exception as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        result["ended_at_utc"]=datetime.now(timezone.utc).isoformat()
        result["disk_free_GiB"]=shutil.disk_usage(ROOT).free/2**30
        save();print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result["status"]=="setup_complete_checkpoint_verified" else 1)


if __name__=="__main__":main()
