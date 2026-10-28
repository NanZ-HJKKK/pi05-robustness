"""Extend the frozen evaluator without modifying the version used for Ref diagnosis."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/"scripts/evaluate.py"
    text=source.read_text(encoding="utf-8")
    original_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    replacements=[
        ('def plan_tasks(kind, suite, trials):','def plan_tasks(kind, suite, trials, plus_split="diagnostic"):'),
        ('        rows = json.loads((ROOT / "configs/eval_subset.json").read_text(encoding="utf-8"))["tasks"]',
         '        filename = "eval_subset.heldout.json" if plus_split == "heldout" else "eval_subset.json"\n        rows = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))["tasks"]'),
        ('    parser.add_argument("--condition", choices=',
         '    parser.add_argument("--plus-split", choices=["diagnostic", "heldout"], default="diagnostic")\n    parser.add_argument("--condition", choices='),
        ('    rows = plan_tasks(args.kind, args.suite, args.trials)',
         '''    if args.kind == "plus":
        registration = json.loads((ROOT/"configs/holdout_registration.json").read_text(encoding="utf-8"))
        assert digest((ROOT/"configs/protocol.json").read_bytes()) == registration["base_protocol_sha256"]
        assert digest((ROOT/"configs/eval_subset.heldout.json").read_bytes()) == registration["heldout_subset_sha256"]
        assert digest((ROOT/"configs/eval_subset.json").read_bytes()) == registration["diagnostic_subset_sha256"]
    rows = plan_tasks(args.kind, args.suite, args.trials, args.plus_split)'''),
        ('        "script_sha256": digest(Path(__file__).read_bytes()),',
         '        "script_sha256": digest(Path(__file__).read_bytes()), "evaluator_filename": Path(__file__).name,'),
        ('            try:\n                for step in range(config["inference"]["max_steps"][suite_name]):',
         '            control_started = time.perf_counter()\n            record["actions"] = []\n            record["rewards"] = []\n            record["done_signals"] = []\n            try:\n                for step in range(config["inference"]["max_steps"][suite_name]):'),
        ('                        timing = dict(response["timing"], request_wall_ms=request_ms, environment_step=step)',
         '''                        timing = dict(response["timing"], request_wall_ms=request_ms, environment_step=step,
                            request_start_elapsed_seconds=infer_start-control_started,
                            request_end_elapsed_seconds=time.perf_counter()-control_started,
                            request_seed_uint32=element["__request_seed"],
                            observed_proprio=element["observation/state"].tolist())'''),
        ('                        infer_start = time.perf_counter()',
         '                        if not np.isfinite(element["observation/state"]).all():\n                            raise ValueError("Nonfinite simulator proprioception")\n                        infer_start = time.perf_counter()'),
        ('                    obs, _, done, _ = env.step(plan.popleft().tolist())',
         '''                    action = plan.popleft().tolist()
                    obs, reward, done, _ = env.step(action)
                    if not np.isfinite(reward):
                        raise ValueError("Nonfinite simulator reward")
                    record["actions"].append(action)
                    record["rewards"].append(float(reward))
                    record["done_signals"].append(bool(done))'''),
        ('            finally:\n                writer.close()',
         '            finally:\n                record["control_loop_wall_seconds"] = time.perf_counter()-control_started\n                writer.close()'),
    ]
    for old,new in replacements:
        assert text.count(old)==1,old
        text=text.replace(old,new,1)
    ast.parse(text)
    destination=ROOT/"scripts/evaluate_v3.py"
    if destination.exists():assert destination.read_text(encoding="utf-8")==text,"Refusing to overwrite changed evaluator v3"
    else:destination.write_text(text,encoding="utf-8",newline="\n")
    receipt={"base_evaluator_sha256":original_sha,"new_evaluator_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
             "changes":["Registered heldout split","Raw applied actions/rewards/done signals","Policy-request seeds and observed proprioception","Simulator control-loop and feedback timestamps"],
             "behavior_unchanged":"Same task resolution, seeding, observations, policy calls and action application; original evaluator stays intact"}
    (ROOT/"evidence/evaluator_v3_source.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
