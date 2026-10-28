"""Reload a LoRA policy and verify finite deterministic actions on a real input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from local_training_config import localize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--num-steps',type=int,choices=[3,10],default=10)
    args=p.parse_args()
    import jax
    import numpy as np
    from openpi.policies import policy_config
    from openpi.training import config
    cfg=localize(config.get_config('pi05_libero_lora_ft'))
    policy=policy_config.create_trained_policy(cfg,args.checkpoint,sample_kwargs={'num_steps':args.num_steps})
    with np.load(args.observation,allow_pickle=False) as data:
        observation={'observation/image':data['main'],'observation/wrist_image':data['wrist'],
            'observation/state':data['state'],'prompt':str(data['prompt'])}
    records=[];outputs=[]
    for seed in [7,7,42]:
        policy._rng=jax.random.key(seed)
        start=time.perf_counter();response=policy.infer(observation)
        actions=np.asarray(response['actions']);jax.block_until_ready(response['actions'])
        assert actions.shape==(10,7) and np.isfinite(actions).all()
        outputs.append(actions.copy())
        records.append({'seed':seed,'shape':list(actions.shape),'min':float(actions.min()),'max':float(actions.max()),
            'sha256':hashlib.sha256(actions.tobytes()).hexdigest(),'wall_seconds':time.perf_counter()-start})
    assert np.array_equal(outputs[0],outputs[1]),'Fixed-seed replay differs'
    result={'passed':True,'checkpoint':str(args.checkpoint),'input_sha256':hashlib.sha256(args.observation.read_bytes()).hexdigest(),
        'num_steps':args.num_steps,'xla_flags':os.environ.get('XLA_FLAGS',''),
        'finite_actions':True,'fixed_seed_replay_equal':True,'records':records,
        'scope':'Checkpoint reload and action-shape/finite/replay check only; not task success or timing benchmark.'}
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
