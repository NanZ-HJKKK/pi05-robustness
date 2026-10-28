"""Verify the candidate against pinned code, including exact global RNG state."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    import numpy as np
    from PIL import Image
    from skimage.filters import gaussian
    from pi05_robustness.fast_glass import glass_blur
    source=ROOT/"third_party/libero_plus/libero/libero/envs/env_wrapper.py"
    text=source.read_text()
    tree=ast.parse(text)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="glass_blur")
    reference_source=ast.get_source_segment(text,function)
    namespace={"np":np,"gaussian":gaussian}
    exec(reference_source,namespace)
    reference=namespace["glass_blur"]
    image=Image.fromarray(np.random.RandomState(123).randint(0,256,(256,256,3),dtype=np.uint8))
    records=[]
    for severity in range(1,11):
        for seed in [0,7,42]:
            np.random.seed(seed);start=time.perf_counter();expected=reference(image,severity);ref_time=time.perf_counter()-start;state_ref=np.random.get_state()
            np.random.seed(seed);start=time.perf_counter();actual=glass_blur(image,severity);new_time=time.perf_counter()-start;state_new=np.random.get_state()
            equal=np.array_equal(expected,actual)
            rng_equal=state_ref[0]==state_new[0] and np.array_equal(state_ref[1],state_new[1]) and state_ref[2:]==state_new[2:]
            records.append({"severity":severity,"seed":seed,"pixels_equal":equal,"rng_state_equal":rng_equal,
                            "reference_seconds":ref_time,"candidate_seconds":new_time})
            if not equal or not rng_equal:
                break
        if not records[-1]["pixels_equal"] or not records[-1]["rng_state_equal"]:break
    result={"passed":len(records)==30 and all(row["pixels_equal"] and row["rng_state_equal"] for row in records),
            "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"reference_function_sha256":hashlib.sha256(reference_source.encode()).hexdigest(),
            "candidate_sha256":hashlib.sha256((ROOT/"src/pi05_robustness/fast_glass.py").read_bytes()).hexdigest(),
            "scope":"10 severities x3 seeds,256RGB; exact float output and MT19937 state. Candidate not yet enabled in any evaluation.","records":records}
    out=ROOT/"runs/fast_glass_equivalence";out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":main()
