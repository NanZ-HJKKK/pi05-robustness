"""Compare Ref actions on identical real input across restarts and source variants."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--expected-model-sha',required=True)
    args=parser.parse_args();assert not args.out.exists();args.out.mkdir(parents=True)
    import jax
    import numpy as np
    import openpi.models.model as model
    from openpi.policies import policy_config
    from openpi.training import config
    actual=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest();assert actual==args.expected_model_sha
    policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),ROOT/'cache/checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
    data=np.load(ROOT/'runs/policy_sanity_input/observation.npz',allow_pickle=False)
    observation={'observation/image':data['main'],'observation/wrist_image':data['wrist'],'observation/state':data['state'],'prompt':str(data['prompt'])}
    row=json.loads((ROOT/'runs/Ref_clean_R5_N10_20261027T040204Z/episodes/libero_spatial/episodes.jsonl').read_text().splitlines()[0])
    seeds=[row['inference'][0]['request_seed_uint32'],7,42]
    outputs={}
    for seed in seeds:
        for repeat in range(2):
            policy._rng=jax.random.key(seed)
            response=policy.infer(observation);actions=np.asarray(response['actions'])
            assert actions.shape==(10,7) and np.isfinite(actions).all()
            outputs[f'{seed}_{repeat}']=actions.copy()
        assert np.array_equal(outputs[f'{seed}_0'],outputs[f'{seed}_1'])
    np.savez_compressed(args.out/'actions.npz',**outputs)
    result={'model_source_file':model.__file__,'model_source_sha256':actual,'gpu_memory_fraction':os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
        'input_sha256':hashlib.sha256((ROOT/'runs/policy_sanity_input/observation.npz').read_bytes()).hexdigest(),
        'within_process_fixed_seed_equal':True,'hashes':{key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in outputs.items()},
        'first_flow_seed':seeds[0],'first5_matches_concurrent_actual':np.array_equal(outputs[f'{seeds[0]}_0'][:5],np.asarray(row['actions'][:5])),
        'first5_max_abs_diff_concurrent_actual':float(np.max(np.abs(outputs[f'{seeds[0]}_0'][:5]-np.asarray(row['actions'][:5])))),
        'scope':'Fixed-input policy repeatability only; not new scored episodes or method gains.'}
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
