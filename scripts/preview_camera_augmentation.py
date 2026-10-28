"""Validate and preview registered augmentation on 16 distinct real training pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from pi05_robustness.augmentation import augment_batch
    data=np.load(args.batch,allow_pickle=False)
    args.out.mkdir(parents=True,exist_ok=True);records=[]
    font=ImageFont.load_default()
    for camera,name,fold in [('main','base_0_rgb',1000),('wrist','left_wrist_0_rgb',1001)]:
        before=(data['image_'+name]+1)/2
        key=jax.random.fold_in(jax.random.key(0),fold)
        after=np.asarray(jax.jit(lambda x:augment_batch(key,x,'view_wrist_drop',camera=camera))(jnp.array(before)))
        assert after.shape==before.shape and np.isfinite(after).all() and after.min()>=0 and after.max()<=1
        unchanged=np.all(before==after,axis=(1,2,3));black=np.all(after==0,axis=(1,2,3))
        assert np.array_equal(np.asarray(augment_batch(key,jnp.array(before),'default',camera=camera)),before)
        sheet=Image.new('RGB',(1792,4*245),'white');draw=ImageDraw.Draw(sheet)
        for i in range(16):
            x,y=(i%4)*448,(i//4)*245;draw.text((x+8,y),camera+' sample '+str(i)+' / original | extra aug',fill='black',font=font)
            for j,arr in enumerate([before[i],after[i]]):
                frame=Image.fromarray(np.uint8(np.clip(arr,0,1)*255)).resize((224,224))
                sheet.paste(frame,(x+j*224,y+20))
        sheet.save(args.out/(camera+'.jpg'),quality=92)
        records.append({'camera':camera,'examples':16,'unchanged':int(unchanged.sum()),'black':int(black.sum()),
            'output_sha256':hashlib.sha256(after.tobytes()).hexdigest(),'finite':True,'default_exact_identity':True})
    result={'passed':True,'seed':0,'batch_sha256':hashlib.sha256(args.batch.read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256((ROOT/'src/pi05_robustness/augmentation.py').read_bytes()).hexdigest(),
        'records':records,'scope':'Extra augmentation on real preprocessed batch; default augmax transforms are applied separately by the trainer. No task success claim.'}
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
