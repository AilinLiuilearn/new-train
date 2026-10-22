"""Index whole train GT components and paired branch-error membership."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.dual_state_maps import MapCache,atomic_json,train_records
from utils.paired_copy_paste import POLICY,raw_record,body_mask,candidates


def build(args):
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    m=json.loads((Path(args.map_dir)/'manifest.json').read_text());cfg=m['baseline_config']
    cache=MapCache(args.map_dir,cfg);records=train_records(cfg,m['train_split_sha256'])
    index={'complete':False,'policy':POLICY,'map_manifest_sha256':cache.identity['manifest_sha256'],
           'image_ids':[r['image_id'] for r in records],'body_bboxes':[],'candidates':[]}
    started=time.monotonic()
    for i,record in enumerate(records):
        ct,pet,gt=raw_record(record,cfg['image_size_2d']);_,bbox=body_mask(ct)
        index['body_bboxes'].append(bbox)
        found=candidates(gt,cache.read(record['image_id'])) if bbox is not None else []
        for item in found:item['record_idx']=i
        index['candidates'].extend(found)
        if i==0 or (i+1)%500==0 or i+1==len(records):
            elapsed=time.monotonic()-started
            print(f'[INDEX] {i+1}/{len(records)} targets={len(index["candidates"])} elapsed={elapsed:.1f}s eta={elapsed/(i+1)*(len(records)-i-1):.1f}s',flush=True)
    index.update(complete=True,elapsed_seconds=time.monotonic()-started,
        counts={'random':len(index['candidates']),
                'hard':sum(item['hard'] for item in index['candidates']),
                'missing_fn':sum(1 in item['flags'] for item in index['candidates']),
                'full_fn':sum(2 in item['flags'] for item in index['candidates'])})
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,index)
    print('[INDEX DONE] '+json.dumps(index['counts']),flush=True)
    return index


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map_dir',required=True);p.add_argument('--output',required=True)
    build(p.parse_args())
