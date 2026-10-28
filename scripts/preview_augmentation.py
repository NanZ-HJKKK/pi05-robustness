"""Preview 16 fixed-seed training augmentations of a real RGB input image."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-kind",choices=["simulator_probe","training_image","unverified_rgb"],default="unverified_rgb")
    args=p.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from pi05_robustness.augmentation import augment_batch
    image=Image.open(args.image).convert("RGB").resize((224,224))
    batch=jnp.asarray(np.repeat(np.asarray(image)[None],16,axis=0),dtype=jnp.float32)/255
    out=np.asarray(jax.jit(lambda x:augment_batch(jax.random.key(0),x,"view_drop"))(batch))
    sheet=Image.new("RGB",(4*224,5*250),(245,245,245))
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.load_default(size=16)
    sheet.paste(image,(0,24));draw.text((5,5),"Original RGB",fill="black",font=font)
    legend=["Proposed extra augmentation / main camera", "Input source: "+args.source_kind,
            "Perspective: each corner displaced by at most 10%", "RGB black drop probability: 15%; fixed seed: 0",
            "Default crop, rotation and color jitter are not shown.","Preview only; no training or success-rate measurement."]
    for i,line in enumerate(legend):draw.text((244,32+i*30),line,fill="black",font=font)
    for i,img in enumerate(out):
        x,y=(i%4)*224,(i//4+1)*250
        draw.text((x+5,y+5),f"seed 0 / sample {i}",fill="black",font=font)
        sheet.paste(Image.fromarray(np.uint8(np.clip(img,0,1)*255)),(x,y+24))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(args.out)
    meta={"input":str(args.image.resolve()),"input_sha256":hashlib.sha256(args.image.read_bytes()).hexdigest(),
          "seed":0,"magnitude":.10,"drop_probability":.15,"input_type":args.source_kind,
          "scope":"extra augmentation only; default augmax transforms not shown; not a training run or performance result"}
    args.out.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
