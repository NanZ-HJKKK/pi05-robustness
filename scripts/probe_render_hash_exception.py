"""Re-render one mismatched initial frame without policy rollout or score changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(array):return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=int,required=True);p.add_argument('--renders',type=int,default=300);args=p.parse_args()
    out=ROOT/'runs/render_hash_exception'/str(args.index);out.mkdir(parents=True)
    source=ROOT/'third_party/libero_plus';package=source/'libero/libero'
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBERO_CONFIG_PATH=str(ROOT/'runs/plus_renderer_probe/libero_config'))
    sys.path.insert(0,str(source))
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    suite=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0);task=suite.get_task(2361)
    assert task.name=='KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_light_4'
    initial=np.asarray(suite.get_task_init_states(2361)[0]);np.random.seed(7)
    env=OffScreenRenderEnv(bddl_file_name=str(package/'bddl_files'/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
    unique={};counts={}
    try:
        env.seed(7);env.reset();obs=env.set_init_state(initial)
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        settled=digest(np.asarray(env.get_sim_state()))
        for repeat in range(args.renders):
            if repeat:obs=env.env._get_observations(force_update=True)
            assert digest(np.asarray(env.get_sim_state()))==settled
            raw=np.ascontiguousarray(obs['agentview_image']);sha=digest(raw);counts[sha]=counts.get(sha,0)+1
            if sha not in unique:
                img=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(raw[::-1,::-1]),224,224))
                np.savez_compressed(out/(sha+'.npz'),raw=raw,model_input=img)
                Image.fromarray(raw).save(out/(sha+'.png'))
                unique[sha]={'model_input_sha256':digest(img),'render_index':repeat}
        report={'process_index':args.index,'renders':args.renders,'settled_state_sha256':settled,'unique':unique,'counts':counts,
            'scope':'Initial-scene repeated renders only; no policy actions or task success evaluation.'}
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    finally:env.close()


if __name__=='__main__':main()
