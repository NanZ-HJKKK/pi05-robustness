"""Freeze task identities and clean prompts from verified upstream text snapshots."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_map(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("libero_task_map assignment missing")


def base_name(name, names):
    matches = [base for base in names if name == base or name.startswith(base + "_")]
    if not matches:
        raise ValueError("No official base task for " + name)
    return max(matches, key=len)


def official_language(name):
    # Same rule as the pinned clean LIBERO grab_language_from_filename.
    if name[0].isupper():
        match = re.search(r"SCENE\d+_(.*)", name)
        if not match: raise ValueError("Missing scene identifier")
        name = match.group(1)
    return " ".join(name.split("_"))


def build():
    upstream = ROOT / "evidence/upstream"
    manifest = json.loads((upstream / "manifest.json").read_text(encoding="utf-8"))
    for repo in manifest.values():
        for item in repo["files"].values():
            assert sha256(upstream / item["file"]) == item["sha256"], item["file"]
    base = read_map(upstream / manifest["libero"]["files"]["libero/libero/benchmark/libero_suite_task_map.py"]["file"])
    plus = read_map(upstream / manifest["libero_plus"]["files"]["libero/libero/benchmark/libero_suite_task_map.py"]["file"])
    catalog = json.loads((upstream / manifest["libero_plus"]["files"]["libero/libero/benchmark/task_classification.json"]["file"]).read_text(encoding="utf-8"))
    prompts, grouped = {}, defaultdict(list)
    for suite in SUITES:
        prompts[suite] = {}
        for name in base[suite]:
            path = f"libero/libero/bddl_files/{suite}/{name}.bddl"
            bddl = upstream / manifest["libero"]["files"][path]["file"]
            match = re.search(r"\(:language\s+(.*?)\)", bddl.read_text(encoding="utf-8"), re.S)
            if not match:
                raise ValueError("Missing BDDL language: " + name)
            prompts[suite][name] = {"prompt": " ".join(match.group(1).strip().strip('"').split()),
                                   "policy_prompt": official_language(name),
                                   "bddl_sha256": sha256(bddl), "task_id": base[suite].index(name)}
        name_to_id = {name: i for i, name in enumerate(plus[suite])}
        for row in catalog[suite]:
            # Classification IDs are 1-based metadata; resolve runtime IDs from names.
            assert row["name"] in name_to_id
            grouped[row["category"]].append({"suite": suite, "task_name": row["name"],
                "task_id": name_to_id[row["name"]], "classification_id": row["id"],
                "category": row["category"], "difficulty": row["difficulty_level"],
                "base_task": base_name(row["name"], base[suite]), "init_state_index": 0})
    assert len(grouped) == 7
    rng = random.Random(0)
    selected = []
    for category, rows in sorted(grouped.items()):
        selected.extend(rng.sample(sorted(rows, key=lambda r: (r["suite"], r["task_name"])), 30))
    assert len({(r["suite"], r["task_name"]) for r in selected}) == 210
    subset = {"seed": 0, "per_category": 30, "source_commit": manifest["libero_plus"]["commit"],
              "selection": "random.Random(0); categories and (suite,name) sorted; pooled four suites", "tasks": selected}
    cfg = ROOT / "configs"
    cfg.mkdir(exist_ok=True)
    for filename, value in (("eval_subset.json", subset), ("base_prompts.json", prompts)):
        (cfg / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "version": 2, "status": "evaluation_fixed_training_pending_resource_probe_and_diagnosis",
        "upstream_commits": {k: v["commit"] for k, v in manifest.items()},
        "subset_sha256": sha256(cfg / "eval_subset.json"), "prompts_sha256": sha256(cfg / "base_prompts.json"),
        "models": ["Ref", "B1", "A1"], "training_seed": 42, "evaluation_seed": 7,
        "clean": {"suites": list(SUITES), "trials_per_task": 10, "episodes_per_model": 400},
        "plus": {"episodes_per_model": 210, "trials_per_task": 1, "prompt_rule": "official clean base task.language except Language Instructions; same canonical prompt as clean evaluation; BDDL text retained for audit"},
        "camera_diagnosis": {"suite": "libero_spatial", "conditions": ["black_main", "black_wrist"], "episodes_per_condition": 100},
        "inference": {"replan_steps": 5, "num_steps": 10, "wait_steps": 10,
                      "flow_noise_seed": "SHA256(eval_seed:suite/base_task/init_state_index:environment_step), first uint32 little-endian; shared across models, camera ablations and perturbation variants",
                      "max_steps": {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}},
        "efficiency": {"suite": "libero_spatial", "episodes_per_setting": 100,
            "settings": [{"replan_steps": r, "num_steps": n} for r, n in [(1,10),(5,10),(10,10),(5,3),(10,3)]],
            "warmup_requests": 5, "latency": "GPU synchronized model sampling and separate request wall time",
            "throughput": "replan_steps / median_sampling_seconds; amortized action throughput, not closed-loop feedback Hz"},
        "training": {"status": "not_started", "steps_proposed": 5000, "batch_proposed": 16,
            "same_B1_A1": ["GPU", "dataset_revision", "initial_checkpoint", "norm_stats", "seed", "batch", "steps", "optimizer", "learning_rate"],
            "augmentation_proposed": "view_drop only if diagnosis supports camera failure; otherwise register a revised strategy before training",
            "checkpoint_rule": "last step; no selection on test subset", "training_seeds": 1,
            "claim_scope": "exploratory single-training-seed study"},
        "acceptance": {"target_A1_minus_B1_pp_min": 0, "clean_drop_pp_max": 2,
                       "note": "estimate with intervals; 30 samples per category cannot establish a small general gain"},
        "budget": {"status": "pending_current_remaining_budget", "automatic_instance_start": False},
        "results": None,
    }
    (cfg / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "categories": Counter(r["category"] for r in selected),
                      "prompts": sum(map(len, prompts.values())), "subset_sha256": protocol["subset_sha256"]}, indent=2))


if __name__ == "__main__":
    build()
