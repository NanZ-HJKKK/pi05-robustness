"""Count parameter shapes without loading a checkpoint or training data."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from flax import nnx
    import jax
    from openpi.training import config
    cfg=config.get_config("pi05_libero_lora_ft")
    shaped=nnx.eval_shape(cfg.model.create,jax.random.key(42))
    params=nnx.state(shaped,nnx.Param)
    trained=params.filter(cfg.trainable_filter)
    def count(tree):return sum(math.prod(x.shape) for x in jax.tree.leaves(tree))
    result={"config":cfg.name,"total_parameters":count(params),"trainable_parameters":count(trained),
            "trainable_fraction":count(trained)/count(params),"batch_size_proposed":cfg.batch_size,
            "scope":"shape inspection only; not memory-fit validation or a parameter update",
            "normalization_assets":str(cfg.data.assets),"pi05":cfg.model.pi05,
            "action_horizon":cfg.model.action_horizon,"discrete_state_input":cfg.model.discrete_state_input}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
