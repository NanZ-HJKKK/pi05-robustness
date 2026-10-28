"""Audit an archived formal clean baseline; never launch model or simulator jobs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi05_robustness.statistics import summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run, archive_sha256):
    archive = run.with_name(run.name + "_evidence.zip")
    assert sha(archive) == archive_sha256
    manifest = read(run / "bundle_manifest.json")
    for item in manifest:
        path = (run / item["path"]).resolve()
        assert path.is_relative_to(run.resolve())
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    snapshot = run / "source_snapshot"
    server = read(run / "server_metadata.json")
    client = read(run / "clean400/run.json")
    result = read(run / "result.json")
    protocol = read(snapshot / "configs/protocol.json")
    prompts = read(snapshot / "configs/base_prompts.json")
    assert sha(snapshot / "scripts/serve.py") == server["server_script_sha256"]
    assert sha(snapshot / "scripts/evaluate.py") == client["script_sha256"]
    assert sha(snapshot / "scripts/run_clean_baseline.py") == result["runner_sha256"]
    assert sha(snapshot / "configs/protocol.json") == client["protocol_sha256"]
    assert sha(snapshot / "configs/base_prompts.json") == protocol["prompts_sha256"]
    assert sha(snapshot / "configs/eval_subset.json") == protocol["subset_sha256"]
    assert sha(run / "checkpoint_verified.json") == server["checkpoint_receipt_sha256"]
    for filename, relative in {
        "model.py": "src/openpi/models/model.py", "policy.py": "src/openpi/policies/policy.py",
        "config.py": "src/openpi/training/config.py",
    }.items():
        assert sha(run / "patched_upstream" / relative) == server["source_sha256"][filename]
    assert result["status"] == "clean400_complete" and result["training_run"] is False
    assert result["owned_processes_stopped"] is True and result["wall_seconds"] <= 10800
    assert result["estimated_compute_during_job_cny"] <= 6.6
    rows = [json.loads(line) for line in (run / "clean400/episodes.jsonl").read_text().splitlines()]
    assert len(rows) == client["expected_episodes"] == 400
    expected = {(suite, name, i) for suite, tasks in prompts.items() for name in tasks for i in range(10)}
    observed = {(row["suite"], row["task_name"], row["init_state_index"]) for row in rows}
    assert expected == observed and len({row["pair_id"] for row in rows}) == 400
    assert all(count == 100 for count in Counter(row["suite"] for row in rows).values())
    for row in rows:
        base = prompts[row["suite"]][row["task_name"]]
        assert row["status"] == "complete" and type(row["success"]) is bool
        assert row["condition"] == "normal" and row["model_id"] == "Ref"
        assert row["prompt"] == base["policy_prompt"] == row["original_task_language"]
        assert row["bddl_sha256"] == base["bddl_sha256"] and row["task_id"] == base["task_id"]
        assert row["replan_steps"] == 5 and row["num_steps"] == 10 and row["seed"] == 7
        assert row["server_run_id"] == server["server_run_id"]
        assert 0 < row["environment_steps"] <= protocol["inference"]["max_steps"][row["suite"]]
        assert len(row["inference"]) == math.ceil(row["environment_steps"] / 5)
        assert [s["environment_step"] for s in row["inference"]] == list(range(0, row["environment_steps"], 5))
        for key in ["initial_state_sha256", "settled_sim_state_sha256", "initial_main_rgb_sha256", "initial_wrist_rgb_sha256"]:
            assert len(row[key]) == 64
        assert (run / "clean400/videos" / Path(row["video"]).name).is_file()
    samples = [sample for row in rows for sample in row["inference"]]
    assert [s["request_index"] for s in samples] == list(range(len(samples)))
    assert [s["warmup"] for s in samples] == [i < 5 for i in range(len(samples))]
    assert all(math.isfinite(s[k]) and s[k] >= 0 for s in samples for k in ["sampling_ms", "policy_wall_ms", "request_wall_ms"])
    summary = summarize(rows)
    reported = read(run / "summary.json")["groups"][0]
    assert all(reported[k] == v for k, v in summary.items())
    assert result["successes"] == summary["successes"]
    assert result["episode_success_rate"] == summary["success_rate"]
    # Smoke is an independent prior run of the same first20 IDs; never add it to400.
    smoke = ROOT / "evidence/phase_one_20261026T164843Z/smoke20/episodes.jsonl"
    prior = [json.loads(line) for line in smoke.read_text().splitlines()]
    by_id = {row["pair_id"]: row for row in rows}
    for old in prior:
        new = by_id[old["pair_id"]]
        for key in ["initial_state_sha256", "settled_sim_state_sha256", "initial_main_rgb_sha256", "initial_wrist_rgb_sha256", "bddl_sha256", "prompt", "seed"]:
            assert old[key] == new[key], "Smoke repeat identity mismatch: " + key
    return {"at_utc": datetime.now(timezone.utc).isoformat(), "verified": True,
            "run_id": run.name, "archive_sha256": archive_sha256, "files_verified": len(manifest),
            "videos_present": len(rows), "all40_tasks_and400_initial_state_ids_present": True,
            "source_hashes_match_runtime_metadata": True, "smoke20_repeat_input_hashes_match": True,
            "summary_recomputed_from_local_raw_rows": summary,
            "per_suite": {suite: summarize([r for r in rows if r["suite"] == suite]) for suite in prompts},
            "scope": "One official checkpoint, one evaluation seed, ten states/task; no training, Plus robustness gain or speedup measured"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.archive_sha256)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
