"""Download pinned public HF files with bounded, resumable ranges and checksums."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def checksums(path, size, check):
    sha = hashlib.sha256()
    blob = hashlib.sha1(("blob %d\0" % size).encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            check()
            sha.update(block)
            blob.update(block)
    return sha.hexdigest(), blob.hexdigest()


def download(url, target, item, check, workers=1, stride=128 * 1024**2):
    size = int(item["size"])
    if size <= 0:
        raise ValueError("Expected a nonempty pinned object")
    expected_sha = item["lfs"]["sha256"] if item.get("lfs") else None
    expected_blob = None if expected_sha else item["oid"]

    def receipt(path):
        sha, blob = checksums(path, size, check)
        if (expected_sha and sha != expected_sha) or (expected_blob and blob != expected_blob):
            raise ValueError("Pinned object checksum mismatch: " + item["path"])
        return {"path": item["path"], "bytes": size, "sha256": sha,
                "git_blob_sha1": blob, "official_sha256": expected_sha}

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError("Unexpected existing object size: " + item["path"])
        return receipt(target)
    chunks = target.with_name(target.name + ".chunks")
    chunks.mkdir(exist_ok=True)
    jobs = [(index, start, min(start + stride, size) - 1)
            for index, start in enumerate(range(0, size, stride))]

    def fetch(job):
        index, start, end = job
        piece = chunks / str(index)
        length = end - start + 1
        if piece.exists():
            if piece.stat().st_size != length:
                raise ValueError("Existing range size mismatch")
            return piece
        partial = chunks / (str(index) + ".partial")
        for attempt in range(8):
            check()
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > length:
                raise ValueError("Partial range exceeds expected length")
            if offset == length:
                partial.replace(piece)
                return piece
            requested = start + offset
            # Unique resolve query avoids intermediary caches reusing another range.
            join = "&" if "?" in url else "?"
            request = urllib.request.Request(url + join + "range_start=" + str(requested),
                headers={"Range": "bytes=%d-%d" % (requested, end), "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    content_range = response.headers.get("Content-Range", "")
                    expected_range = "bytes %d-%d/%d" % (requested, end, size)
                    whole = requested == 0 and end == size - 1 and response.status == 200
                    if not whole and (response.status != 206 or content_range != expected_range):
                        raise ValueError("Range response does not match requested bytes")
                    received = offset
                    with partial.open("ab" if offset else "wb") as stream:
                        while True:
                            check()
                            block = response.read(1024**2)
                            if not block:
                                break
                            received += len(block)
                            if received > length:
                                raise ValueError("Response exceeds expected range")
                            stream.write(block)
                if received != length:
                    raise ConnectionError("Truncated body; resume received prefix")
                partial.replace(piece)
                return piece
            except Exception as exc:
                check()
                if isinstance(exc, ValueError) or attempt == 7:
                    raise
                time.sleep(min(2**attempt, 10))
        raise RuntimeError("Unreachable range retry state")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pieces = list(pool.map(fetch, jobs))
    staging = target.with_name(target.name + ".verified_staging")
    with staging.open("wb") as stream:
        for piece in pieces:
            check()
            with piece.open("rb") as source:
                shutil.copyfileobj(source, stream, 1024**2)
    if staging.stat().st_size != size:
        raise ValueError("Assembled size mismatch")
    result = receipt(staging)
    staging.replace(target)
    for piece in pieces:
        piece.unlink()
    chunks.rmdir()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["dataset", "plus_assets"], required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.max_hours <= 0 or args.workers < 1:
        parser.error("Positive limits required")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((ROOT / "evidence/resource_inventory.json").read_text())[args.kind]
    records = []
    mutex = threading.Lock()
    cancelled = threading.Event()
    first_error = []
    start = time.monotonic()
    deadline = start + args.max_hours * 3600
    out = ROOT / "runs" / (args.kind + "_download")
    out.mkdir(parents=True, exist_ok=True)

    def check():
        if cancelled.is_set():
            raise RuntimeError("Sibling transfer stopped; partial bytes retained")
        if time.monotonic() >= deadline:
            raise TimeoutError("Download window ended; ranges retained")
        if shutil.disk_usage(destination).free < 2 * 1024**3:
            raise RuntimeError("Less than2GiB free on destination filesystem")

    def transfer(item):
        try:
            target = (destination / item["path"]).resolve()
            target.relative_to(destination)
            url = "https://huggingface.co/datasets/%s/resolve/%s/%s" % (
                inventory["repo"], inventory["revision"], urllib.parse.quote(item["path"], safe="/"))
            result = download(url, target, item, check,
                              workers=24 if args.kind == "plus_assets" else 1,
                              stride=16 * 1024**2 if args.kind == "plus_assets" else 128 * 1024**2)
            with mutex:
                records.append(result)
                with (out / "verified_files.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps({"verified": item["path"], "completed": len(records),
                                  "total": len(inventory["files"]), "bytes": item["size"]}), flush=True)
            return result
        except BaseException as exc:
            with mutex:
                if not first_error:
                    detail = {"file": item["path"], "error_type": type(exc).__name__,
                              "http_status": getattr(exc, "code", None)}
                    if isinstance(exc, (ValueError, RuntimeError)):
                        detail["detail"] = str(exc)[:180]
                    first_error.append(detail)
                    print(json.dumps({"first_error": detail}), flush=True)
                cancelled.set()
            raise

    report = {"repo": inventory["repo"], "revision": inventory["revision"],
              "destination": str(destination), "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        with (destination / ".resource_transfer.lock").open("a") as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with ThreadPoolExecutor(max_workers=1 if args.kind == "plus_assets" else args.workers) as pool:
                list(pool.map(transfer, inventory["files"]))
        records.sort(key=lambda row: row["path"])
        manifest = out / "verified_manifest.json"
        manifest.write_text(json.dumps({"repo": inventory["repo"], "revision": inventory["revision"],
                                       "files": records}, indent=2) + "\n")
        report.update(status="complete_verified", files=len(records), bytes=sum(row["bytes"] for row in records),
                      manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    except Exception as exc:
        # Avoid logging redirect URLs or proxy credentials in transport errors.
        report.update(status="incomplete", error_type=type(exc).__name__,
                      http_status=getattr(exc, "code", None), verified_files=len(records),
                      first_error=first_error[0] if first_error else None)
    finally:
        report.update(wall_seconds=time.monotonic() - start, ended_at_utc=datetime.now(timezone.utc).isoformat())
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "complete_verified" else 1)


if __name__ == "__main__":
    main()
