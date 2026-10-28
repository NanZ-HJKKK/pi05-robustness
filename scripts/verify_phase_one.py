"""Independently verify the downloaded smoke evidence without starting any jobs."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi05_robustness.statistics import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    run_id = "phase_one_20261026T164843Z"
    run = ROOT / "evidence" / run_id
    archive = ROOT / "evidence" / (run_id + "_evidence.zip")
    expected_archive = "03b3dab4d76c9d24259fc79ff74d12386360205ee9d44bcc655c86830fba6415"
    assert digest(archive) == expected_archive
    manifest = json.loads((run / "bundle_manifest.json").read_text())
    for item in manifest:
        path = (run / item["path"]).resolve()
        assert path.is_relative_to(run.resolve())
        assert path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"]

    server = json.loads((run / "server_metadata.json").read_text())
    client = json.loads((run / "smoke20/run.json").read_text())
    result = json.loads((run / "result.json").read_text())
    snapshot = run / "source_snapshot"
    assert digest(snapshot / "scripts/serve.py") == server["server_script_sha256"]
    assert digest(snapshot / "scripts/evaluate.py") == client["script_sha256"]
    assert digest(snapshot / "configs/protocol.json") == client["protocol_sha256"]
    assert digest(run / "setup_receipts/checkpoint_verified.json") == server["checkpoint_receipt_sha256"]
    for name, relative in {
        "model.py": "src/openpi/models/model.py",
        "policy.py": "src/openpi/policies/policy.py",
        "config.py": "src/openpi/training/config.py",
    }.items():
        assert digest(run / "patched_upstream" / relative) == server["source_sha256"][name]

    rows = [json.loads(line) for line in (run / "smoke20/episodes.jsonl").read_text().splitlines()]
    assert result["status"] == "smoke_complete" and result["training_run"] is False
    assert len(rows) == client["expected_episodes"] == 20
    assert all(row["status"] == "complete" and row["success"] is True for row in rows)
    assert len({row["pair_id"] for row in rows}) == 20
    assert len({row["base_task"] for row in rows}) == 2
    assert all(row["suite"] == "libero_spatial" for row in rows)
    for row in rows:
        assert row["model_id"] == "Ref" and row["condition"] == "normal"
        assert row["replan_steps"] == 5 and row["num_steps"] == 10 and row["seed"] == 7
        assert row["server_run_id"] == server["server_run_id"]
        assert len(row["inference"]) == math.ceil(row["environment_steps"] / 5)
        assert [s["environment_step"] for s in row["inference"]] == list(range(0, row["environment_steps"], 5))
        assert (run / "smoke20/videos" / Path(row["video"]).name).is_file()
    samples = [sample for row in rows for sample in row["inference"]]
    assert [sample["request_index"] for sample in samples] == list(range(391))
    assert [sample["warmup"] for sample in samples] == [i < 5 for i in range(391)]
    assert all(math.isfinite(s[k]) and s[k] >= 0 for s in samples for k in ["sampling_ms", "policy_wall_ms", "request_wall_ms"])
    probe = json.loads((ROOT / "evidence/simulator_probe/result.json").read_text())
    assert rows[0]["settled_sim_state_sha256"] == probe["sim_state_sha256"]
    recomputed = summarize(rows)
    reported = json.loads((run / "summary.json").read_text())["groups"][0]
    assert all(reported[key] == value for key, value in recomputed.items())
    return {
        "at_utc": datetime.now(timezone.utc).isoformat(), "verified": True,
        "run_id": run_id, "archive_sha256": expected_archive,
        "files_verified": len(manifest), "videos_present": 20,
        "source_hashes_match_runtime_metadata": True,
        "request_indices_contiguous": True, "initial_probe_state_matches": True,
        "summary_recomputed_from_local_raw_rows": recomputed,
        "episode_wall_seconds_mean_excluding_first": statistics.mean(r["episode_wall_seconds"] for r in rows[1:]),
        "scope": "First two Spatial tasks only; no training or robustness/optimization gain measured",
    }


if __name__ == "__main__":
    output = verify()
    (ROOT / "evidence/phase_one_acceptance.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
