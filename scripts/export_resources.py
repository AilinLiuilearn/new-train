"""Prepare an optional private transfer bundle; no Git or cloud upload occurs."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import REPO, local_paths, verify_resources, sha


def export_bundle(locations, output, include_data=False):
    output = Path(output).expanduser().resolve()
    if output.exists():raise FileExistsError(output)
    files = []
    for name in ['convnextv2_nano', 'mit-b1']:
        root = Path(locations['pretrained_root']) / name
        for p in sorted(root.rglob('*')):
            if p.is_file():files.append((p, 'pretrained/' + name + '/' + str(p.relative_to(root))))
    if include_data:
        from datasets.pclt20k_seg import _read_list, _records_from_ids
        ids = sorted(set(_read_list(REPO / 'splits/train.txt') + _read_list(REPO / 'splits/dev.txt')))
        root = Path(locations['data_root'])
        for rec in _records_from_ids(str(root), ids):
            for key in ['ct_path', 'pet_path', 'mask_path']:
                p = Path(rec[key]);files.append((p, 'data/PCLT20K/' + str(p.relative_to(root))))
    entries = {name: {'bytes':p.stat().st_size, 'sha256':sha(p)} for p,name in files}
    record = {'format':'ct_pet_resources_v1','include_data':include_data,'files':entries,
              'note':'Private transfer only; image data and model weights do not belong in Git'}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    try:
        with tarfile.open(temporary, 'w:gz', compresslevel=1) as tf:
            for p,name in files:tf.add(p,arcname=name,recursive=False)
            payload=(json.dumps(record,indent=2)+'\n').encode()
            info=tarfile.TarInfo('RESOURCE_MANIFEST.json');info.size=len(payload)
            tf.addfile(info,io.BytesIO(payload))
        temporary.rename(output)
    finally:
        if temporary.exists():temporary.unlink()
    return {'output':str(output),'files':len(entries),'archive_sha256':sha(output)}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--paths',default='paths.local.json')
    p.add_argument('--output',required=True)
    p.add_argument('--include-data',action='store_true')
    args=p.parse_args();locations=local_paths(args.paths)
    verify_resources(locations)
    print(json.dumps(export_bundle(locations,args.output,args.include_data),indent=2))
