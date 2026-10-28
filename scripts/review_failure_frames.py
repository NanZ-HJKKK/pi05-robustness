"""Extract fixed timeline frames for every failure of a diagnostic category."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=Path,required=True)
    p.add_argument('--category',default='Camera Viewpoints')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    rows=[json.loads(x) for x in (args.episodes/'episodes.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['status']=='complete' and not r['success'] and r['category']==args.category]
    args.out.mkdir(parents=True,exist_ok=True);records=[]
    font=ImageFont.load_default()
    for i,row in enumerate(rows):
        video=Path(row['video']);reader=imageio.get_reader(str(video))
        try:
            count=reader.count_frames();indices=sorted(set(int(x) for x in np.linspace(0,count-1,6)))
            sheet=Image.new('RGB',(896,120+len(indices)//2*250),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,8),row['suite']+' / '+row['base_task'],fill='black',font=font)
            draw.text((8,30),row['requested_bddl_name'],fill='black',font=font)
            draw.text((8,54),'LEFT: main camera; RIGHT: wrist camera / failed at '+str(row['environment_steps'])+' steps',fill='black',font=font)
            for j,index in enumerate(indices):
                frame=Image.fromarray(reader.get_data(index));frame.thumbnail((448,224))
                x,y=(j%2)*448,120+(j//2)*250;sheet.paste(frame,(x,y+20))
                draw.text((x+8,y),'frame '+str(index)+' / '+str(count),fill='black',font=font)
            target=args.out/(str(i).zfill(2)+'.jpg');sheet.save(target,quality=92)
            records.append({'pair_id':row['pair_id'],'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
                'video':str(video),'frames':count,'sampled_indices':indices,'sheet':str(target),'label':'','notes':''})
        finally:reader.close()
    (args.out/'review.json').write_text(json.dumps({'category':args.category,'records':records},indent=2)+'\n')
    print('Extracted',len(records),'failure timelines')


if __name__=='__main__':main()
