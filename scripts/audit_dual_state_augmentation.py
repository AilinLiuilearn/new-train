"""Fixed-seed train-only visual/quantitative audit; never consumes training RNG."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from datasets.pclt20k_seg import _imread_grayscale, _resize_gray
from utils.dual_state_maps import MapCache, atomic_json, train_records, sha256
from utils.dual_state_augmentation import augment, summarize

PALETTE = np.array([[0,0,0], [255,80,40], [40,160,255], [200,60,220]], dtype=np.uint8)


def panel(array, label, window=None):
    array = np.asarray(array, dtype=np.uint8)
    im = Image.fromarray(array).convert('RGB')
    if window:
        x,y,w,h = window
        ImageDraw.Draw(im).rectangle((x,y,x+w-1,y+h-1), outline='yellow', width=3)
    im = im.resize((256,256))
    result = Image.new('RGB', (256,280))
    result.paste(im,(0,24)); ImageDraw.Draw(result).text((4,5),label,fill='white')
    return result


def audit(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new audit output directory')
    if args.num_samples < 1:
        raise ValueError('num_samples must be positive')
    if not 0 <= args.guided_crop_prob <= 1 or args.visual_samples < 0:
        raise ValueError('Invalid guidance probability/visual count')
    manifest_path = Path(args.map_dir)/'manifest.json'
    m = json.loads(manifest_path.read_text())
    cfg = m['baseline_config']
    cache = MapCache(args.map_dir, cfg)
    records = train_records(cfg, m['train_split_sha256'])
    # Random sample plus explicit coverage examples. Report selection design.
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(records)).tolist()
    n = min(args.num_samples,len(records))
    strata = {}
    for idx in order:
        sample = m['samples'][records[idx]['image_id']]
        p = sample['pixels']
        for category in (1,2,3):
            if p[category]: strata.setdefault(f'class_{category}',idx)
        if p[1] and p[2]: strata.setdefault('both',idx)
        if not p[1]+p[2]: strata.setdefault('no_differential',idx)
        if sample.get('background_false_positive_regions',0): strata.setdefault('background_false_positive',idx)
    selected = list(dict.fromkeys(list(strata.values())+order))[:n]
    output.mkdir(parents=True)
    totals, rows, events = {}, [], {}
    print(f'[AUDIT] start samples={n} strategy={args.crop_guidance} seed={args.seed}',flush=True)
    for position, idx in enumerate(selected):
        r = records[idx]
        size = cfg['image_size_2d']
        ct, pet, gt = [_imread_grayscale(r[key]) for key in ('ct_path','pet_path','mask_path')]
        if gt.max() <= 1: gt = gt*np.uint8(255)
        if ct.shape != (size,size):
            ct,pet,gt = _resize_gray(ct,size), _resize_gray(pet,size), _resize_gray(gt,size,nearest=True)
        d = cache.read(r['image_id'])
        np.random.seed(args.seed+position)  # Independent from run and sample-selection RNG.
        image, mask, cropped_d, info, trace, before = augment(np.stack([ct,pet],-1),gt,d,
                args.crop_guidance,args.guided_crop_prob,return_trace=True,
                region_sampling=args.guidance_region_sampling)
        for key,value in info.items(): totals[key] = totals.get(key,0.)+value
        row = {'image_id':r['image_id'],'seed':args.seed+position,**info,**trace,
               'post_mask_binary_values':np.unique((mask >= 127.5).astype(np.uint8)).tolist()}
        rows.append(row)
        event = ('fallback' if info['guided_attempt'] and not info['guided_success'] else
                 'guided' if info['guided_success'] else 'random' if info['crop_triggered'] else 'no_crop')
        events.setdefault(event, r['image_id'])
        save_visual = position < args.visual_samples or events[event] == r['image_id']
        if save_visual:
            bi,bm,bd = before
            window, random_window = trace['window'],trace['random_window']
            random_image,random_mask = bi,bm
            if random_window:
                x,y,w,h = random_window
                random_image = cv2.resize(bi[y:y+h,x:x+w],(size,size),interpolation=cv2.INTER_CUBIC)
                random_mask = cv2.resize(bm[y:y+h,x:x+w],(size,size),interpolation=cv2.INTER_CUBIC)
            tiles = [panel(ct,'original CT'),panel(pet,'original PET'),panel(gt,'original GT'),panel(PALETTE[d],'D: 1 red / 2 blue / 3 purple'),
                     panel(bi[...,0],'transformed CT / chosen',window),panel(bi[...,1],'transformed PET / random',random_window),
                     panel(bm,'transformed GT / chosen',window),panel(PALETTE[bd],'transformed D / chosen',window),
                     panel(image[...,0],'chosen crop CT'),panel(image[...,1],'chosen crop PET'),
                     panel((mask >=127.5).astype(np.uint8)*255,'chosen crop binary GT'),panel(PALETTE[cropped_d],'chosen crop D'),
                     panel(random_image[...,0],'same-size random CT'),panel(random_image[...,1],'same-size random PET'),
                     panel((random_mask >=127.5).astype(np.uint8)*255,'same-size random GT'),panel(PALETTE[bd],'D without crop')]
            canvas = Image.new('RGB',(1024,1120))
            for j,tile in enumerate(tiles): canvas.paste(tile,((j%4)*256,(j//4)*280))
            canvas.save(output/(r['image_id']+'.png'))
        if position == 0 or (position+1)%50 == 0 or position+1 == n:
            print(f'[AUDIT] {position+1}/{n}',flush=True)
    summary = summarize(totals)
    warnings=[]
    coverage=m['coverage']
    if coverage['no_differential']/m['count'] > .5:
        warnings.append('More than half of train samples have no differential candidates.')
    if summary['success_per_attempt'] < .5:
        warnings.append('More than half of guidance attempts fall back; inspect no-candidate/target-size counts.')
    if summary['mean_window_iou'] > .8:
        warnings.append('Large crops produce strongly overlapping windows; inspect incremental view change.')
    if 'fallback' not in events:
        warnings.append('No real fallback visual in this sample; synthetic boundary tests provide coverage.')
    report={'seed':args.seed,'selection':'fixed random order with available class/background strata prepended',
        'sample_count':n,'strata':{k:records[v]['image_id'] for k,v in strata.items()},
        'manifest_sha256':sha256(manifest_path),'strategy':args.crop_guidance,
        'guided_crop_prob':args.guided_crop_prob,'train_coverage':coverage,
        'region_sampling':args.guidance_region_sampling,
        'no_differential_fraction':coverage['no_differential']/m['count'],
        'both_differential_fraction':coverage['both_differential']/m['count'],
        'augmentation':summary,'warnings':warnings,'event_examples':events}
    atomic_json(output/'summary.json',report)
    (output/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    print('[AUDIT DONE] '+json.dumps(report),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map_dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--num_samples',type=int,default=32)
    p.add_argument('--visual_samples',type=int,default=32)
    p.add_argument('--seed',type=int,default=2023)
    p.add_argument('--crop_guidance',choices=['foreground','hard','dual_state'],default='dual_state')
    p.add_argument('--guided_crop_prob',type=float,default=.5)
    p.add_argument('--guidance_region_sampling',choices=['uniform','sqrt_area'],default='uniform')
    return p


if __name__=='__main__':
    audit(parser().parse_args())
