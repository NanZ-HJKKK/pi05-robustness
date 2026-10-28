"""Resumable HTTP ranges; verify official size, CRC32C and available MD5."""
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))


def download_checkpoint(items, cache, check, workers=32):
    cache=Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    with (cache/".checkpoint_transfer.lock").open("a") as lock:
        if os.name=="posix":
            import fcntl
            while True:
                check()
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:time.sleep(2)
        try:return _download_checkpoint(items,cache,check,workers)
        finally:
            if os.name=="posix":fcntl.flock(lock,fcntl.LOCK_UN)


def _download_checkpoint(items, cache, check, workers=32):
    cache=Path(cache)
    verified=[]
    for item in items:
        name=item["name"]
        if not name.startswith("checkpoints/pi05_libero/"):raise ValueError("Unexpected checkpoint prefix")
        target=cache/name
        if not target.resolve().is_relative_to(cache.resolve()):raise ValueError("Path escapes cache")
        size=int(item["size"])
        if not item.get("md5Hash") and not item.get("crc32c"):
            raise ValueError("Official object has no integrity checksum: "+name)
        def checksum(path):
            md5=hashlib.md5();sha=hashlib.sha256()
            crc=None
            if item.get("crc32c"):
                import google_crc32c
                crc=google_crc32c.Checksum()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(8*1024**2),b""):
                    check();md5.update(block);sha.update(block)
                    if crc is not None:crc.update(block)
            return base64.b64encode(md5.digest()).decode(),sha.hexdigest(),base64.b64encode(crc.digest()).decode() if crc else None
        def matches(md5,crc):
            return (not item.get("md5Hash") or md5==item["md5Hash"]) and (not item.get("crc32c") or crc==item["crc32c"])
        if target.exists() and target.stat().st_size==size:
            md5,sha,crc=checksum(target)
            if matches(md5,crc):
                verified.append({"name":name,"size":size,"sha256":sha,"crc32c":crc});continue
        target.parent.mkdir(parents=True,exist_ok=True)
        chunks=target.parent/(target.name+".chunks")
        chunks.mkdir(exist_ok=True)
        stride=16*1024**2
        jobs=[(i,start,min(start+stride,size)-1) for i,start in enumerate(range(0,size,stride))]

        def fetch(job):
            index,start,end=job
            piece=chunks/str(index)
            if piece.exists() and piece.stat().st_size==end-start+1:return piece
            temporary=chunks/(str(index)+".partial")
            for attempt in range(8):
                check()
                offset=temporary.stat().st_size if temporary.exists() else 0
                if offset>end-start+1:raise ValueError("Partial range exceeds expected size")
                if offset==end-start+1:
                    temporary.replace(piece)
                    return piece
                requested_start=start+offset
                endpoint="https://storage.googleapis.com/openpi-assets/" if attempt%2==0 else "https://openpi-assets.storage.googleapis.com/"
                req=urllib.request.Request(endpoint+name,
                    headers={"Range":f"bytes={requested_start}-{end}"})
                try:
                    with urllib.request.urlopen(req,timeout=30) as response:
                        if response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {requested_start}-{end}/"):
                            raise ValueError("Range response mismatch")
                        received=offset
                        with temporary.open("ab" if offset else "wb") as stream:
                            while True:
                                check();block=response.read(1024**2)
                                if not block:break
                                received+=len(block)
                                if received>end-start+1:raise ValueError("Range exceeds expected size")
                                stream.write(block)
                    if received!=end-start+1:raise ConnectionError("Range truncated; preserve received bytes and retry")
                    temporary.replace(piece)
                    return piece
                except Exception as exc:
                    check()
                    if isinstance(exc,ValueError) or attempt==7:raise
                    time.sleep(min(2**attempt,10))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pieces=list(pool.map(fetch,jobs))
        part=target.with_name(target.name+".part")
        with part.open("wb") as out:
            for piece in pieces:
                check()
                with piece.open("rb") as src:shutil.copyfileobj(src,out,1024**2)
        md5,sha,crc=checksum(part)
        if part.stat().st_size!=size or not matches(md5,crc):raise ValueError("Checkpoint integrity failed: "+name)
        part.replace(target)
        for piece in pieces:piece.unlink()
        chunks.rmdir()
        verified.append({"name":name,"size":size,"sha256":sha,"crc32c":crc})
        print(f"verified {name} {size} bytes",flush=True)
    return verified


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-minutes",type=float,default=30)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    started=time.monotonic();deadline=started+args.max_minutes*60
    result={"status":"running","max_minutes":args.max_minutes}
    def check():
        if time.monotonic()>deadline:raise TimeoutError("Checkpoint transfer window ended; chunks preserved")
        if shutil.disk_usage(root).free<2*2**30:raise RuntimeError("Less than 2GiB disk free")
    try:
        inventory=json.loads((root/"evidence/resource_inventory.json").read_text())
        verified=download_checkpoint(inventory["checkpoint"]["files"],root/"cache",check)
        result.update(status="verified",files=verified)
    except Exception as exc:result.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    result["wall_seconds"]=time.monotonic()-started
    (root/"runs/checkpoint_transfer.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result["status"]=="verified" else 1)


if __name__=="__main__":main()
