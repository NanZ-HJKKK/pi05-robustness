"""Archive a completed clean400 run and extract video samples; no model loading."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to((ROOT / "runs").resolve())
    result = json.loads((run / "result.json").read_text())
    if result["status"] != "clean400_complete" or not result["owned_processes_stopped"]:
        raise SystemExit("Refusing to archive an unfinished run as a full baseline")
    rows = [json.loads(line) for line in (run / "clean400/episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 400 and all(row["status"] == "complete" for row in rows)
    assert len({row["pair_id"] for row in rows}) == 400
    assert set(Counter(row["suite"] for row in rows).values()) == {100}
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), str(run / "clean400"),
                    "--out", str(run / "summary.json")], check=True, stdout=subprocess.DEVNULL)
    failures = [{k: row[k] for k in ["pair_id", "suite", "task_name", "init_state_index", "environment_steps", "video"]}
                for row in rows if not row["success"]]
    (run / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    selected_successes = []
    for suite in dict.fromkeys(row["suite"] for row in rows):
        selected_successes.extend([i for i, row in enumerate(rows) if row["suite"] == suite and row["success"]][:1])
    selected_failures = [i for i, row in enumerate(rows) if not row["success"]][:16]
    checks = []
    for kind, selected in [("success", selected_successes), ("failure", selected_failures)]:
        for page_start in range(0, len(selected), 4):
            page_rows = selected[page_start:page_start + 4]
            sheet = Image.new("RGB", (1344, 256 * len(page_rows)), "white")
            draw = ImageDraw.Draw(sheet)
            for j, index in enumerate(page_rows):
                row = rows[index]
                path = run / "clean400/videos" / Path(row["video"]).name
                reader = imageio.get_reader(str(path))
                count = reader.count_frames()
                assert count == row["environment_steps"]
                frame_indices = [0, count // 2, count - 1]
                for k, frame_index in enumerate(frame_indices):
                    im = Image.fromarray(reader.get_data(frame_index)).resize((448, 224))
                    sheet.paste(im, (448 * k, 256 * j + 28))
                    draw.text((448 * k + 5, 256 * j + 7),
                              "%s ep%d %s frame%d/%d" % (row["suite"], index, kind, frame_index, count), fill="black")
                reader.close()
                checks.append({"row_index": index, "pair_id": row["pair_id"], "success": row["success"],
                               "frames": count, "frame_indices": frame_indices})
            sheet.save(run / ("video_%s_samples_%d.jpg" % (kind, page_start // 4 + 1)), quality=92)
    (run / "video_frames_checked.json").write_text(json.dumps(checks, indent=2) + "\n")
    post = run / "postprocessing_source"
    for relative in ["scripts/bundle_clean_baseline.py", "scripts/verify_clean_baseline.py",
                     "scripts/summarize.py", "src/pi05_robustness/statistics.py"]:
        dst = post / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dst)
    manifest = [{"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": digest(path)}
                for path in sorted(run.rglob("*")) if path.is_file() and path.name != "bundle_manifest.json"]
    (run / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = run.with_name(run.name + "_evidence.zip")
    with zipfile.ZipFile(str(archive), "w") as z:
        for path in sorted(run.rglob("*")):
            if path.is_file():
                z.write(str(path), str(path.relative_to(run.parent)),
                        compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED)
    receipt = {"at_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
               "archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest(archive),
               "payload_files": len(manifest), "videos": len(list((run / "clean400/videos").glob("*.mp4"))),
               "sampled_videos": len(checks), "failed_episodes": len(failures)}
    archive.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
