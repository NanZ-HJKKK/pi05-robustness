"""One bounded renderer/reset probe; no policy and no success-rate measurement."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"third_party/openpi/third_party/libero"))


def main():
    out=ROOT/"runs/simulator_probe"
    out.mkdir(parents=True,exist_ok=True)
    cfg=out/"libero_config";cfg.mkdir(exist_ok=True)
    package=ROOT/"third_party/openpi/third_party/libero/libero/libero"
    values={"benchmark_root":str(package),"bddl_files":str(package/"bddl_files"),
            "init_states":str(package/"init_files"),"assets":str(package/"assets"),"datasets":str(ROOT/"cache/libero_datasets")}
    (cfg/"config.yaml").write_text(json.dumps(values))
    os.environ.update(LIBERO_CONFIG_PATH=str(cfg),MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",OMP_NUM_THREADS="4")
    result={"passed":False,"policy_used":False,"success_rate":None,"evaluation_episodes":0,
            "suite":"libero_spatial","task_id":0,"seed":7}
    start=time.monotonic();env=None
    try:
        import numpy as np
        from PIL import Image
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools
        suite=benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
        task=suite.get_task(0)
        expected=json.loads((ROOT/"configs/base_prompts.json").read_text())["libero_spatial"][task.name]
        assert expected["policy_prompt"]==task.language
        bddl=package/"bddl_files"/task.problem_folder/task.bddl_file
        assert hashlib.sha256(bddl.read_bytes()).hexdigest()==expected["bddl_sha256"]
        states=suite.get_task_init_states(0)
        np.random.seed(7)
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.seed(7);env.reset();obs=env.set_init_state(states[0])
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        for key,name in [("agentview_image","input_main.png"),("robot0_eye_in_hand_image","input_wrist.png")]:
            array=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[key][::-1,::-1]),224,224))
            assert array.shape==(224,224,3) and np.isfinite(array).all() and array.std()>1
            Image.fromarray(array).save(out/name)
            result[key]={"shape":list(array.shape),"dtype":str(array.dtype),"sha256":hashlib.sha256(array.tobytes()).hexdigest()}
        state=np.asarray(env.get_sim_state())
        assert np.isfinite(state).all()
        result.update(passed=True,task_name=task.name,prompt=task.language,
                      sim_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                      dummy_settling_steps=10,orientation="rotate180 then resize224 as official client")
    except Exception:
        result["error"]=traceback.format_exc()
    finally:
        if env is not None:
            try:env.close()
            except Exception:
                result.update(passed=False,cleanup_error=traceback.format_exc())
        result["wall_seconds"]=time.monotonic()-start
        (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":main()
