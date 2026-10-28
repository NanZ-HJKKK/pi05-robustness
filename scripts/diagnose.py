"""Rank Ref failure dimensions with task-matched clean controls; create review CSV."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics


def load(paths,expected_kind):
    rows=[]
    for path in paths:
        attempts=[json.loads(line) for line in (path/"episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        complete=[r for r in attempts if r["status"]=="complete"]
        metadata=json.loads((path/"run.json").read_text(encoding="utf-8"))
        if metadata["settings"]["kind"]!=expected_kind:raise ValueError("Wrong run kind for diagnosis")
        if expected_kind=="plus" and metadata["settings"].get("plus_split","diagnostic")!="diagnostic":
            raise ValueError("Heldout outcomes cannot be used for augmentation selection")
        expected=metadata["expected_episodes"]
        if len(complete)!=expected:raise ValueError("Resolve incomplete episodes before diagnosis")
        rows.extend(complete)
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate episodes")
    if any(r["model_id"] != "Ref" or r["condition"] != "normal" or r["replan_steps"] != 5 or r["num_steps"] != 10 for r in rows):
        raise ValueError("Diagnosis needs only normal Ref with default inference settings")
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean",nargs="+",type=Path,required=True)
    p.add_argument("--plus",nargs="+",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    clean,plus=load(args.clean,"clean"),load(args.plus,"plus")
    by_task,categories=defaultdict(list),defaultdict(list)
    for r in clean: by_task[(r["suite"],r["base_task"])].append(int(r["success"]))
    exact_init={(r["suite"],r["base_task"],r["init_state_index"]):r for r in clean}
    for r in plus: categories[r["category"]].append(r)
    ranking=[]
    for category,rows in categories.items():
        control=statistics.mean(statistics.mean(by_task[(r["suite"],r["base_task"])]) for r in rows)
        perturbed=statistics.mean(int(r["success"]) for r in rows)
        matched=[(r,exact_init[(r["suite"],r["base_task"],r["init_state_index"])]) for r in rows]
        initial_control=statistics.mean(int(c["success"]) for r,c in matched)
        ranking.append({"category":category,"episodes":len(rows),"success_rate":perturbed,
            "task_mix_matched_clean_rate":control,"change_pp":100*(perturbed-control),
            "additional_same_index_clean_rate":initial_control,
            "additional_same_index_change_pp":100*(perturbed-initial_control),
            "raw_initial_vector_hash_matches":sum(r["initial_state_sha256"]==c["initial_state_sha256"] for r,c in matched),
            "settled_sim_state_hash_matches":sum(r["settled_sim_state_sha256"]==c["settled_sim_state_sha256"] for r,c in matched)})
    ranking.sort(key=lambda r:r["change_pp"])
    visual={"Camera Viewpoints","Light Conditions","Background Textures","Sensor Noise"}
    worst=next(r for r in ranking if r["category"] in visual)
    target=worst["category"] if worst["change_pp"]<0 else None
    failures=sorted([r for r in categories[target] if not r["success"]],key=lambda r:r["pair_id"]) if target else []
    sample=random.Random(0).sample(failures,min(20,len(failures)))
    args.out.mkdir(parents=True,exist_ok=True)
    report={"ranking":ranking,"provisional_visual_target":target,"review_count":len(sample),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "additional_control_scope":"Same-index control is a descriptive sensitivity analysis; repeated base controls are not independent, physical/layout changes may alter state. Primary ranking remains task-mix matched rate.",
        "caution":"Task-mix adjusted descriptive drop; not identical physical states or a causal attribution. Confirm failure video labels and camera ablation before choosing training enhancement."}
    (args.out/"diagnosis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (args.out/"failure_review.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=["pair_id","video","label","notes"])
        writer.writeheader()
        for r in sample:writer.writerow({"pair_id":r["pair_id"],"video":r.get("video"),"label":"","notes":""})
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
