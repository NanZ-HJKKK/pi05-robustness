"""Build the offline Arrow cache and verify a real CPU training batch; no model."""
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    start=time.monotonic()
    out=ROOT/"runs/training_data_preflight"
    out.mkdir(parents=True,exist_ok=True)
    report={"status":"running","model_loaded":False,"training_steps":0,
            "started_at_utc":datetime.now(timezone.utc).isoformat()}
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    os.environ.update(JAX_PLATFORMS="cpu",HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",
        HF_LEROBOT_HOME=str(ROOT/"cache/training"),HF_DATASETS_CACHE="/dev/shm/pi05_arrow",
        OPENPI_DATA_HOME=str(ROOT/"cache/openpi"),TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4")
    try:
        receipt=json.loads((ROOT/"runs/dataset_download/result.json").read_text())
        assert receipt["status"]=="complete_verified"
        import jax
        import numpy as np
        from PIL import Image
        from openpi.training import config,data_loader
        from local_training_config import localize
        cfg=dataclasses.replace(localize(config.get_config("pi05_libero_lora_ft")),batch_size=16,num_workers=0)
        loader=data_loader.create_data_loader(cfg,shuffle=True,num_batches=1)
        observation,actions=next(iter(loader))
        assert list(actions.shape)==[16,10,32]
        assert np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(observation.state)).all()
        images={}
        for key,value in observation.images.items():
            array=np.asarray(value)
            assert array.shape==(16,224,224,3) and np.isfinite(array).all()
            active=bool(np.asarray(observation.image_masks[key])[0])
            images[key]={"shape":list(array.shape),"min":float(array.min()),"max":float(array.max()),"active":active,
                         "batch_sha256":hashlib.sha256(array.tobytes()).hexdigest()}
            if active:
                for index in range(16):
                    rgb=np.clip((array[index]+1)*127.5,0,255).astype(np.uint8)
                    Image.fromarray(rgb).save(out/("%s_%02d.png"%(key,index)))
        snapshot={"state":np.asarray(observation.state),"actions":np.asarray(actions)}
        for key,value in observation.images.items():snapshot["image_"+key]=np.asarray(value)
        for key,value in observation.image_masks.items():snapshot["mask_"+key]=np.asarray(value)
        for key in ["tokenized_prompt","tokenized_prompt_mask"]:
            value=getattr(observation,key)
            if value is not None:snapshot[key]=np.asarray(value)
        np.savez_compressed(out/"real_first_batch.npz",**snapshot)
        metadata=json.loads((ROOT/"cache/training/physical-intelligence/libero/meta/info.json").read_text())
        report.update(status="passed",jax_devices=[str(x) for x in jax.devices()],
            dataset_revision=receipt["revision"],dataset_manifest_sha256=receipt["manifest_sha256"],
            dataset_episodes=metadata["total_episodes"],dataset_frames=metadata["total_frames"],dataset_tasks=metadata["total_tasks"],
            action_shape=list(actions.shape),state_shape=list(observation.state.shape),images=images,
            batch_snapshot_sha256=hashlib.sha256((out/"real_first_batch.npz").read_bytes()).hexdigest(),
            scope="16 real normalized training inputs before model default/extra augmentation; no training or performance result")
    except Exception:
        report.update(status="failed",error=traceback.format_exc())
    finally:
        report["wall_seconds"]=time.monotonic()-start
        (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if report["status"]=="passed" else 1)


if __name__=="__main__":main()
