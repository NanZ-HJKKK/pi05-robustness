"""Read-only resource check; no imports of GPU frameworks or resource downloads."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path("/root/autodl-tmp"))
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    report={"at_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),"python":sys.version,
            "data_root":str(args.data_root),"linux":platform.system()=="Linux",
            "tools":{name:shutil.which(name) for name in ["uv","git","nvidia-smi"]}}
    if args.data_root.exists():
        report["free_GiB"]=round(shutil.disk_usage(args.data_root).free/2**30,2)
    report["packages"]={}
    for name in ["jax","jaxlib","torch","lerobot","mujoco","robosuite"]:
        try: report["packages"][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report["packages"][name]=None
    if shutil.which("nvidia-smi"):
        gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free,driver_version","--format=csv,noheader"],capture_output=True,text=True,timeout=15)
        report["gpu"]={"exit_code":gpu.returncode,"stdout":gpu.stdout,"stderr":gpu.stderr}
    report["result"]="inventory_only; training fit and simulator runtime unverified"
    text=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text,encoding="utf-8")
    print(text)


if __name__=="__main__":
    main()
