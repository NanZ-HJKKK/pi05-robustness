"""Register a disjoint Plus holdout before perturbation diagnosis is observed."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from prepare_protocol import read_map, base_name, SUITES

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = ROOT / "configs"
    manifest = json.loads((ROOT / "evidence/upstream/manifest.json").read_text())
    upstream = ROOT / "evidence/upstream"
    source = manifest["libero_plus"]
    for entry in source["files"].values():
        assert sha(upstream / entry["file"]) == entry["sha256"]
    base = json.loads((cfg / "base_prompts.json").read_text())
    task_map = read_map(upstream / source["files"]["libero/libero/benchmark/libero_suite_task_map.py"]["file"])
    catalog = json.loads((upstream / source["files"]["libero/libero/benchmark/task_classification.json"]["file"]).read_text())
    diagnostic = json.loads((cfg / "eval_subset.json").read_text())
    excluded = {(row["suite"], row["task_name"]) for row in diagnostic["tasks"]}
    grouped = defaultdict(list)
    for suite in SUITES:
        ids = {name: i for i, name in enumerate(task_map[suite])}
        for row in catalog[suite]:
            if (suite, row["name"]) in excluded:
                continue
            grouped[row["category"]].append({"suite": suite, "task_name": row["name"],
                "task_id": ids[row["name"]], "classification_id": row["id"],
                "category": row["category"], "difficulty": row["difficulty_level"],
                "base_task": base_name(row["name"], base[suite]), "init_state_index": 0})
    rng = random.Random(1)
    selected = []
    for category, rows in sorted(grouped.items()):
        selected.extend(rng.sample(sorted(rows, key=lambda row: (row["suite"], row["task_name"])), 30))
    ids = {(row["suite"], row["task_name"]) for row in selected}
    assert len(ids) == 210 and not ids.intersection(excluded)
    assert set(Counter(row["category"] for row in selected).values()) == {30}
    subset = {"seed": 1, "per_category": 30, "source_commit": source["commit"],
              "selection": "random.Random(1); category and(suite,name) sorted; exclude all diagnostic210 exact task IDs",
              "role": "Final heldout evaluation; no augmentation, hyperparameter or checkpoint selection",
              "tasks": selected}
    path = cfg / "eval_subset.heldout.json"
    serialized = json.dumps(subset, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text(encoding="utf-8") == serialized, "Refusing to alter a registered holdout"
    else:
        path.write_text(serialized, encoding="utf-8")
    registration = cfg / "holdout_registration.json"
    if not registration.exists():
        value = {"version": 1, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                 "stage": "Before any Ref Plus outcome; clean baseline complete and camera diagnosis running",
                 "base_protocol_sha256": sha(cfg / "protocol.json"),
                 "diagnostic_subset_sha256": sha(cfg / "eval_subset.json"),
                 "heldout_subset_sha256": sha(path), "disjoint_exact_task_ids": True,
                 "diagnostic_episodes": 210, "heldout_episodes_per_model": 210,
                 "selection_rule": "Diagnose Ref on diagnostic210 and camera ablation; freeze training plan before viewing heldout outcomes",
                 "models_on_holdout": ["Ref", "B1", "A1"],
                 "other_settings": "Base protocol unchanged, including canonical prompts, seed7, initial-state index0 and flow-noise rule",
                 "planned_formal_episodes": 2740,
                 "limits": "Same40 base tasks; exact variants disjoint, not unseen-task generalization. One training seed remains exploratory."}
        registration.write_text(json.dumps(value, indent=2) + "\n")
    value = json.loads(registration.read_text())
    assert value["heldout_subset_sha256"] == sha(path)
    assert value["diagnostic_subset_sha256"] == sha(cfg / "eval_subset.json")
    print(json.dumps({"heldout": len(selected), "disjoint": True,
                      "heldout_sha256": sha(path), "registration_sha256": sha(registration)}, indent=2))


if __name__ == "__main__":
    main()
