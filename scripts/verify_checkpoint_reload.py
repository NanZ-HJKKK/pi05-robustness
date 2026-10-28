"""Restore original-dtype parameters on CPU and match final training fingerprints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--audit-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    from flax.traverse_util import flatten_dict
    from openpi.models.model import restore_params
    reference=json.loads((args.audit_dir/"params_final.json").read_text())
    params=flatten_dict(restore_params(args.checkpoint/"params",restore_type=np.ndarray))
    checked=0
    for row in reference["parameters"]:
        value=np.asarray(params[tuple(row["path"])])
        assert list(value.shape)==row["shape"] and str(value.dtype)==row["dtype"],row["path"]
        sample=value.reshape(-1)[:row["sample_count"]]
        assert hashlib.sha256(sample.tobytes()).hexdigest()==row["sample_sha256"],row["path"]
        checked+=1
    proof=json.loads((args.audit_dir/"update_proof.json").read_text())
    assert proof["passed"]
    result={"at_utc":datetime.now(timezone.utc).isoformat(),"passed":True,"checkpoint":str(args.checkpoint),
            "restored_parameter_leaves_sampled":checked,"sample_rule":reference["sample_rule"],
            "training_step":reference["training_step"],"training_update_proof_passed":True,
            "scope":"CPU original-dtype reload matches final training parameter samples; action output verified separately"}
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
