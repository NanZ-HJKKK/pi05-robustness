"""Prepare pinned LIBERO-Plus in an independent client, leaving clean untouched."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.monotonic()
    deadline = start + 7200
    run = ROOT / "runs/setup_plus"
    run.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(), "steps": []}
    output = run / "result.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "tools"), UV_CACHE_DIR=str(ROOT / "cache/uv"),
               UV_HTTP_TIMEOUT="60", UV_CONCURRENT_DOWNLOADS="8", GIT_LFS_SKIP_SMUDGE="1")
    env["no_proxy"] = env.get("no_proxy", "") + ",mirrors.aliyun.com"
    uv = [sys.executable, "-m", "uv"]

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def check():
        if time.monotonic() > deadline:
            raise TimeoutError("Plus setup window ended")
        if shutil.disk_usage(ROOT).free < 2 * 1024**3:
            raise RuntimeError("Less than2GiB data-disk space")

    def command(name, argv):
        check()
        report["steps"].append({"name": name, "argv": argv})
        save()
        with (run / (name + ".log")).open("w") as log:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=max(1, deadline - time.monotonic()))
        report["steps"][-1]["exit_code"] = result.returncode
        save()
        if result.returncode:
            raise RuntimeError(name + " failed; inspect its log")

    try:
        lock = json.loads((ROOT / "evidence/upstream/manifest.json").read_text())["libero_plus"]
        source = ROOT / "cache/libero_plus_source.zip"
        if not source.exists():
            partial = source.with_suffix(".zip.partial")
            url = "https://codeload.github.com/%s/zip/%s" % (lock["repo"], lock["commit"])
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                total = 0
                while True:
                    check()
                    block = response.read(1024**2)
                    if not block:
                        break
                    total += len(block)
                    if total > 1536 * 1024**2:
                        raise ValueError("Source archive larger than expected cap")
                    stream.write(block)
            partial.replace(source)
        repo = ROOT / "third_party/libero_plus"
        if not (repo / ".snapshot_complete").exists():
            repo.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(source) as archive:
                for item in archive.infolist():
                    check()
                    relative = Path(*Path(item.filename).parts[1:])
                    if relative.parts[:3] == ("libero", "libero", "assets"):
                        continue  # Separate verified official Plus assets supply this tree.
                    destination = (repo / relative).resolve()
                    destination.relative_to(repo.resolve())
                    if item.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(item) as src, destination.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            (repo / ".snapshot_complete").write_text(lock["commit"] + "\n")
        for name, entry in lock["files"].items():
            if hashlib.sha256((repo / name).read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Pinned Plus source mismatch: " + name)
        report["source_archive_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        report["source_commit"] = lock["commit"]
        frozen = subprocess.run(uv + ["pip", "freeze", "--python", str(ROOT / ".venv_clean/bin/python")],
                                env=env, capture_output=True, text=True, check=True, timeout=60).stdout
        # Editable clean LIBERO is the only package to replace, with Plus below.
        lines = [line for line in frozen.splitlines() if line and not line.startswith("-e ")]
        requirements = run / "clean_pins.requirements.txt"
        requirements.write_text("\n".join(lines) + "\n")
        clean_hash = hashlib.sha256((ROOT / ".venv_clean/pyvenv.cfg").read_bytes()).hexdigest()
        plus = ROOT / ".venv_plus"
        if not plus.exists():
            command("venv", uv + ["venv", "--python", "3.8", str(plus)])
        command("core_packages", uv + ["pip", "install", "--python", str(plus / "bin/python"),
            "-r", str(requirements), "--index-url", "https://mirrors.aliyun.com/pypi/simple", "--torch-backend", "cpu"])
        command("plus_packages", uv + ["pip", "install", "--python", str(plus / "bin/python"),
            "--no-deps", "-e", str(repo), "-e", str(ROOT / "third_party/openpi/packages/openpi-client")])
        command("plus_extras", uv + ["pip", "install", "--python", str(plus / "bin/python"),
            "-c", str(requirements), "Wand==0.6.13", "scikit-image==0.21.0",
            "--index-url", "https://mirrors.aliyun.com/pypi/simple"])
        if "libMagickWand" not in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout:
            command("magick_runtime", ["apt-get", "install", "-y", "--no-install-recommends", "libmagickwand-6.q16-6"])
        command("pip_check", uv + ["pip", "check", "--python", str(plus / "bin/python")])
        command("runtime_import", [str(plus / "bin/python"), "-c",
            "import wand.image,skimage,numpy,torch; print('Wand',wand.version.VERSION,'skimage',skimage.__version__,'numpy',numpy.__version__,'torch',torch.__version__)"])
        assert clean_hash == hashlib.sha256((ROOT / ".venv_clean/pyvenv.cfg").read_bytes()).hexdigest()
        report.update(status="plus_environment_ready_assets_separate", clean_environment_config_unchanged=True)
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc)[:200])
    finally:
        report.update(wall_seconds=time.monotonic() - start, ended_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "plus_environment_ready_assets_separate" else 1)


if __name__ == "__main__":
    main()
