"""Enable a verified equivalent implementation only while the Plus client is stopped."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    proof=json.loads((ROOT/"runs/fast_glass_equivalence/result.json").read_text())
    real=json.loads((ROOT/"runs/fast_glass_equivalence/real_frame_check.json").read_text())
    assert proof["passed"] and len(proof["records"])==30 and real["passed"] and len(real["cases"])==10
    source=ROOT/"third_party/libero_plus/libero/libero/envs/env_wrapper.py"
    helper=ROOT/"src/pi05_robustness/fast_glass.py"
    assert hashlib.sha256(helper.read_bytes()).hexdigest()==proof["candidate_sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==proof["source_sha256"]
    backup=source.with_name('env_wrapper.reference.py');assert not backup.exists();shutil.copy2(source,backup)
    destination=source.with_name('exact_glass.py');shutil.copy2(helper,destination)
    text=source.read_text()+'''\n# Exact output/RNG equivalence proof is archived in this experiment.\nif os.environ.get("LIBERO_EXACT_GLASS", "1") == "1":\n    from .exact_glass import glass_blur as glass_blur\n'''
    source.write_text(text)
    result={"reference_source_sha256":proof["source_sha256"],
            "patched_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "helper_sha256":proof["candidate_sha256"],"synthetic_checks":30,"real_frame_checks":10,
            "pixels_and_numpy_rng_state_equal":True,
            "scope":"Environment glass-corruption execution only; policy and task/noise sampling unchanged. Not a model inference speedup."}
    (ROOT/"evidence/exact_glass_activation.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
