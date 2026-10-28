"""Verify and extract pinned Plus runtime assets onto the separate system disk."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path("/root/pi05_storage")
ARCHIVE_PREFIX = "inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/"


def main():
    started = time.monotonic()
    run = ROOT / "runs/plus_assets_extract"
    run.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    output = run / "result.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        download = json.loads((ROOT / "runs/plus_assets_download/result.json").read_text())
        assert download["status"] == "complete_verified"
        item = json.loads((ROOT / "evidence/resource_inventory.json").read_text())["plus_assets"]["files"][0]
        source = STORAGE / "downloads/assets.zip"
        assert source.stat().st_size == item["size"]
        h = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(block)
        assert h.hexdigest() == item["lfs"]["sha256"]
        runtime_files = 0
        runtime_bytes = 0
        skipped_metadata = 0
        with zipfile.ZipFile(source) as archive, (run / "files.jsonl").open("w") as inventory:
            for index, member in enumerate(archive.infolist()):
                if not member.filename.startswith(ARCHIVE_PREFIX):
                    raise ValueError("Unexpected archive prefix: " + member.filename)
                relative = Path(member.filename[len(ARCHIVE_PREFIX):])
                destination = (STORAGE / relative).resolve()
                destination.relative_to(STORAGE.resolve())
                if relative.parts and relative.parts[0] == "__MACOSX":
                    skipped_metadata += 1
                    continue
                if not relative.parts or relative.parts[0] != "assets":
                    raise ValueError("Unexpected ZIP root: " + member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Inspect asset symlink before extraction: " + member.filename)
                if index % 512 == 0:
                    if time.monotonic() - started > 7200:
                        raise TimeoutError("Extraction window ended; completed files retained")
                    if shutil.disk_usage(STORAGE).free < 2 * 1024**3:
                        raise RuntimeError("Less than2GiB system disk free")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                existing_valid = False
                digest = hashlib.sha256()
                if destination.exists() and destination.stat().st_size == member.file_size:
                    crc = 0
                    with destination.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024**2), b""):
                            digest.update(block)
                            crc = zlib.crc32(block, crc)
                    existing_valid = (crc & 0xffffffff) == member.CRC
                if not existing_valid:
                    digest = hashlib.sha256()
                    temporary = destination.with_name(destination.name + ".extracting")
                    with archive.open(member) as src, temporary.open("wb") as dst:
                        while True:
                            block = src.read(1024**2)
                            if not block:
                                break
                            digest.update(block)
                            dst.write(block)
                    # ZipExtFile validates CRC32 while fully reading each member.
                    assert temporary.stat().st_size == member.file_size
                    temporary.replace(destination)
                record = {"path": str(relative), "archive_path": member.filename, "bytes": member.file_size,
                          "zip_crc32": member.CRC, "sha256": digest.hexdigest()}
                inventory.write(json.dumps(record) + "\n")
                runtime_files += 1
                runtime_bytes += member.file_size
                if runtime_files % 5000 == 0:
                    inventory.flush()
                    print(json.dumps({"files": runtime_files, "bytes": runtime_bytes}), flush=True)
        assets = ROOT / "third_party/libero_plus/libero/libero/assets"
        target = STORAGE / "assets"
        if assets.is_symlink():
            assert assets.resolve() == target.resolve()
        elif assets.exists():
            raise ValueError("Refusing to replace a pre-existing Plus assets directory")
        else:
            assets.symlink_to(target, target_is_directory=True)
        manifest = run / "files.jsonl"
        h = hashlib.sha256()
        with manifest.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(block)
        report.update(status="complete_verified", source_sha256=item["lfs"]["sha256"],
                      source_revision=download["revision"], runtime_files=runtime_files,
                      runtime_bytes=runtime_bytes, skipped_macos_metadata_entries=skipped_metadata,
                      file_manifest_sha256=h.hexdigest(), stripped_archive_prefix=ARCHIVE_PREFIX,
                      assets=str(target), link=str(assets))
    except Exception as exc:
        report.update(status="incomplete", error=type(exc).__name__ + ": " + str(exc)[:220])
    finally:
        report.update(wall_seconds=time.monotonic() - started, ended_at_utc=datetime.now(timezone.utc).isoformat())
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "complete_verified" else 1)


if __name__ == "__main__":
    main()
