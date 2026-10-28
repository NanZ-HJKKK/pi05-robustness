"""Invoke the pinned trainer with local verified files and explicit run settings."""
import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys

from local_training_config import localize

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config",choices=["pi05_libero_lora_ft"])
    p.add_argument("--exp-name",required=True)
    p.add_argument("--batch-size",type=int,required=True)
    p.add_argument("--num-train-steps",type=int,required=True)
    p.add_argument("--checkpoint-base-dir",required=True)
    p.add_argument("--keep-period",type=int,default=1000000000)
    args=p.parse_args()
    sys.path.insert(0,str(ROOT/"third_party/openpi/scripts"))
    from openpi.training import config
    import train
    cfg=localize(config.get_config(args.config))
    cfg=dataclasses.replace(cfg,exp_name=args.exp_name,batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,checkpoint_base_dir=args.checkpoint_base_dir,
        keep_period=args.keep_period)
    if out:=os.environ.get("OPENPI_AUDIT_DIR"):
        Path(out,"effective_config.json").write_text(json.dumps({
            "name":cfg.name,"batch_size":cfg.batch_size,"num_train_steps":cfg.num_train_steps,
            "seed":cfg.seed,"assets_dir":cfg.data.assets.assets_dir,"weight_loader":repr(cfg.weight_loader),
            "checkpoint_dir":str(cfg.checkpoint_dir),"dataset_repo":cfg.data.repo_id,
            "augmentation":os.environ.get("OPENPI_AUG","default"),
            "local_paths_only":True,"stats_recomputed":False},indent=2)+"\n")
    logging.basicConfig(level=logging.INFO)
    train.main(cfg)


if __name__=="__main__":main()
