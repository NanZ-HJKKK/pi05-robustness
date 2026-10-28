"""Capture a fixed real simulator input; no policy rollout or scored episode."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate import quat_to_axisangle

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    row=next(json.loads(line) for line in (args.baseline/'episodes.jsonl').read_text().splitlines())
    suite=benchmark.get_benchmark_dict()[row['suite']](task_order_index=0)
    task=suite.get_task(row['task_id'])
    state=np.asarray(suite.get_task_init_states(row['task_id'])[row['init_state_index']])
    assert task.name==row['task_name']
    np.random.seed(row['seed'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
    try:
        env.seed(row['seed']);env.reset();obs=env.set_init_state(state)
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        hashes={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in {
            'settled_sim_state_sha256':np.asarray(env.get_sim_state()),
            'initial_main_rgb_sha256':np.ascontiguousarray(obs['agentview_image']),
            'initial_wrist_rgb_sha256':np.ascontiguousarray(obs['robot0_eye_in_hand_image'])}.items()}
        assert all(row[k]==v for k,v in hashes.items()),hashes
        images=[image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[k][::-1,::-1]),224,224)) for k in ['agentview_image','robot0_eye_in_hand_image']]
        proprio=np.concatenate([obs['robot0_eef_pos'],quat_to_axisangle(obs['robot0_eef_quat'],np),obs['robot0_gripper_qpos']])
        args.out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.out,main=images[0],wrist=images[1],state=proprio,prompt=np.array(row['prompt']))
        result={'passed':True,'baseline_pair_id':row['pair_id'],'hashes':hashes,'npz_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),'scope':'Input capture only; no scored policy episode.'}
        args.out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':main()
