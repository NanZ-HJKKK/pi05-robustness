"""Summarize measured run directories; refuse incomplete runs by default."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pi05_robustness.statistics import summarize, paired_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true")
    args = parser.parse_args()
    groups,errors,coverage = defaultdict(list),[],[]
    for run in args.runs:
        meta = json.loads((run/"run.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (run/"episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        complete = [r for r in records if r["status"] == "complete"]
        errors.extend({"run":str(run),"pair_id":r["pair_id"],"status":r["status"],"error":r.get("error")} for r in records if r["status"] != "complete")
        if len(complete) != meta["expected_episodes"] and not args.allow_partial:
            raise SystemExit("Incomplete run: " + str(run))
        coverage.append({"run":str(run),"expected":meta["expected_episodes"],"completed":len(complete)})
        settings = meta["settings"]
        for row in complete:
            split=settings.get("plus_split","diagnostic") if settings["kind"]=="plus" else "clean"
            key = (row["model_id"],settings["kind"],row["condition"],row["category"],row["replan_steps"],row["num_steps"],split)
            groups[key].append(row)
    report = {"status":"partial" if args.allow_partial else "complete_input_runs", "coverage":coverage,"error_attempts":errors,"groups":[],"paired_A1_vs_B1":[]}
    for key,rows in sorted(groups.items()):
        report["groups"].append({"model":key[0],"kind":key[1],"condition":key[2],"category":key[3],
            "replan_steps":key[4],"num_steps":key[5],"split":key[6],**summarize(rows)})
        if key[0] == "A1" and ("B1",)+key[1:] in groups:
            report["paired_A1_vs_B1"].append({"group":list(key[1:]),**paired_comparison(groups[("B1",)+key[1:]],rows)})
    report["scope"] = "Only supplied runs; this does not certify the full protocol, failure review, or GitHub publication. Single training seed is exploratory."
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
