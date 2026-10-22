"""Fixed train-only paired paste audit, including original/ donor/ mixed views."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from utils.paired_copy_paste import PasteBank,raw_record,summarize
from utils.dual_state_maps import atomic_json


def tile(a,label,box=None):
    im=Image.fromarray(a).convert('RGB')
    if box:ImageDraw.Draw(im).rectangle((box[0],box[1],box[2]-1,box[3]-1),outline='yellow',width=2)
    im=im.resize((256,256));out=Image.new('RGB',(256,280));out.paste(im,(0,24))
    ImageDraw.Draw(out).text((4,5),label,fill='white');return out


def audit(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((Path(args.map_dir)/'manifest.json').read_text());cfg=manifest['baseline_config']
    bank=PasteBank(args.paste_index,args.map_dir,cfg,selection=args.paste_selection)
    output.mkdir(parents=True)
    ids=np.random.RandomState(args.seed).permutation(len(bank.records))[:args.num_samples]
    totals={};rows=[];visuals=0;started=time.monotonic()
    for j,idx in enumerate(ids):
        original=raw_record(bank.records[idx],bank.size)
        np.random.seed(args.seed+j)
        ct,pet,gt,info,trace=bank.paste(*original,int(idx),args.mode,1.,trace=True)
        for key,value in info.items():totals[key]=totals.get(key,0.)+value
        row={'recipient_id':bank.records[idx]['image_id'],'seed':args.seed+j,**info,**trace};rows.append(row)
        if info['success']:
            x0,y0,x1,y1=trace['recipient_bbox'];dx0,dy0,dx1,dy1=trace['donor_bbox']
            donor_idx=next(i for i,r in enumerate(bank.records) if r['image_id']==trace['donor_id'])
            donor=bank.read(donor_idx)
            for mixed,source in zip((ct,pet,gt),donor):
                if not np.array_equal(mixed[y0:y1,x0:x1],source[dy0:dy1,dx0:dx1]):raise AssertionError('Paste pairing mismatch')
            if np.any(original[2][y0:y1,x0:x1]>0):raise AssertionError('Overwrote recipient GT')
            if visuals<args.visual_samples:
                tiles=[]
                for arrays,title,box in [(original,'recipient',trace['recipient_bbox']),
                                         (donor,'donor',trace['donor_bbox']),((ct,pet,gt),'mixed',trace['recipient_bbox'])]:
                    for array,channel in zip(arrays,['CT','PET','GT']):tiles.append(tile(array,title+' '+channel,box))
                for arrays,title,box in [(original,'recipient patch',trace['recipient_bbox']),
                                         (donor,'donor patch',trace['donor_bbox']),((ct,pet,gt),'mixed patch',trace['recipient_bbox'])]:
                    ax,ay,bx,by=box
                    pad=max(8,round(.2*max(bx-ax,by-ay)))
                    ax,ay,bx,by=max(0,ax-pad),max(0,ay-pad),min(bank.size,bx+pad),min(bank.size,by+pad)
                    for array,channel in zip(arrays,['CT','PET','GT']):
                        tiles.append(tile(array[ay:by,ax:bx],title+' '+channel))
                canvas=Image.new('RGB',(768,1680))
                for k,im in enumerate(tiles):canvas.paste(im,((k%3)*256,(k//3)*280))
                canvas.save(output/f'{j:04d}-{trace["category"]}-{bank.records[idx]["image_id"]}.png');visuals+=1
        if j==0 or (j+1)%100==0 or j+1==len(ids):
            print(f'[PASTE AUDIT] {j+1}/{len(ids)} success={totals.get("success",0)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    summary={'mode':args.mode,'seed':args.seed,'forced_attempt_probability':1.,'training_probability':.3,
             'expected_training_success_rate':.3*totals.get('success',0)/max(len(ids),1),
             'counts':summarize(totals),'visuals':visuals,'identity':bank.identity,
             'warning':'Body-relative placement and CT context are heuristics, not organ/anatomical verification.'}
    atomic_json(output/'summary.json',summary)
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('[PASTE AUDIT DONE] '+json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map_dir',required=True);p.add_argument('--paste_index',required=True)
    p.add_argument('--output',required=True);p.add_argument('--mode',choices=['random','hard','dual_state'],default='dual_state')
    p.add_argument('--num_samples',type=int,default=512);p.add_argument('--visual_samples',type=int,default=24)
    p.add_argument('--seed',type=int,default=2023)
    p.add_argument('--paste_selection',choices=['overlap','dominance'],default='overlap')
    audit(p.parse_args())
