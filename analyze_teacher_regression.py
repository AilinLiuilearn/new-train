# -*- coding: utf-8 -*-
import argparse, csv, json, os
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from models.build_mdt_seg import build_mdt_seg_teacher
from utils.metrics_seg import compute_hd95_pair


def parse_args():
    p = argparse.ArgumentParser('Analyze new teacher regression vs baseline teacher')
    p.add_argument('--baseline_dir', required=True)
    p.add_argument('--new_dir', required=True)
    p.add_argument('--split', default='test', choices=('val', 'test'))
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out_dir', default=None)
    return p.parse_args()


def read_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def ckpt_path(d):
    for n in ('ckpt.best_dice.pth.tar', 'ckpt.best.pth.tar', 'ckpt.best_hd95.pth.tar', 'ckpt.last.pth.tar'):
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(d)


def clean_state(ckpt):
    if isinstance(ckpt, dict):
        for k in ('ema_model', 'model', 'state_dict'):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    return {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}


def load_teacher(run_dir, device):
    cfg = SimpleNamespace(**read_json(os.path.join(run_dir, 'config_args.json')))
    model = build_mdt_seg_teacher(cfg)['model']
    path = ckpt_path(run_dir)
    msg = model.load_state_dict(clean_state(torch.load(path, map_location='cpu', weights_only=False)), strict=False)
    report = {
        'run_dir': run_dir, 'ckpt': path,
        'num_missing': len(msg.missing_keys), 'num_unexpected': len(msg.unexpected_keys),
        'missing_head': list(msg.missing_keys)[:40],
        'unexpected_head': list(msg.unexpected_keys)[:40],
    }
    return cfg, model.to(device).eval(), report


def dataset_module():
    import importlib.util
    root = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def loaders(cfg, bs, nw):
    m = dataset_module()
    if getattr(cfg, 'cipa_aligned', False):
        return m.get_pclt20k_loaders_cipa_aligned(cfg.root, cfg.image_size_2d, bs, nw, cfg.random_state, pin_memory=getattr(cfg, 'pin_memory', True))
    return m.get_pclt20k_loaders(cfg.root, cfg.image_size_2d, bs, nw, val_ratio=cfg.val_ratio, random_state=cfg.random_state, use_case_split=getattr(cfg, 'use_case_split', True), pin_memory=getattr(cfg, 'pin_memory', True))


def div(a, b):
    return 0.0 if b == 0 else float(a) / float(b)


def metric(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp, fp, fn = np.logical_and(p, g).sum(), np.logical_and(p, ~g).sum(), np.logical_and(~p, g).sum()
    dice = 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return dict(dice=float(dice), iou=float(iou), hd95=float(compute_hd95_pair(p, g)), recall=div(tp, tp + fn), precision=div(tp, tp + fp), fp=int(fp), fn=int(fn), pred_area=int(p.sum()), gt_area=int(g.sum()))


def desc_cos(a, b):
    a = F.adaptive_avg_pool2d(a.float(), 1).flatten(1)
    b = F.adaptive_avg_pool2d(b.float(), 1).flatten(1)
    return float(F.cosine_similarity(a, b, dim=1).mean().detach().cpu())


def aux_stat(out):
    res = {}
    auxs = out.get('disentangle_aux') if isinstance(out, dict) else None
    if not auxs:
        return res
    for i, a in enumerate(auxs, 1):
        pre = f's{i}'
        res[f'{pre}_common_cos'] = desc_cos(a['ct_common'], a['pet_common'])
        res[f'{pre}_specific_cos'] = desc_cos(a['ct_specific'], a['pet_specific'])
        for k in ('ct_common', 'pet_common', 'ct_specific', 'pet_specific'):
            res[f'{pre}_{k}_abs'] = float(a[k].float().abs().mean().detach().cpu())
    return res


def add_q(rows, key, name, labels):
    vals = np.array([r[key] for r in rows], dtype=np.float32)
    qs = [np.quantile(vals, i / len(labels)) for i in range(1, len(labels))]
    for r in rows:
        idx = 0
        while idx < len(qs) and r[key] > qs[idx]:
            idx += 1
        r[name] = labels[idx]


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else 0.0


def write_group(path, rows, gkey):
    keys = ['base_dice', 'new_dice', 'delta_dice', 'base_hd95', 'new_hd95', 'delta_hd95', 'gt_area', 'base_area', 'new_area', 'new_recall', 'new_precision', 'new_fp', 'new_fn', 'pet_contrast', 'ct_contrast']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow([gkey, 'count'] + keys)
        for g in sorted(set(r[gkey] for r in rows)):
            sub = [r for r in rows if r[gkey] == g]
            w.writerow([g, len(sub)] + [f'{mean(sub, k):.4f}' for k in keys])


def main():
    args = parse_args(); device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    base_cfg, base, br = load_teacher(args.baseline_dir, device)
    new_cfg, new, nr = load_teacher(args.new_dir, device)
    _, vl, tl = loaders(new_cfg, args.batch_size, args.num_workers)
    loader = tl if args.split == 'test' else vl
    out_dir = args.out_dir or os.path.join(args.new_dir, f'analysis_vs_{os.path.basename(args.baseline_dir)}_{args.split}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'load_report.json'), 'w') as f: json.dump({'baseline': br, 'new': nr}, f, indent=2)
    rows, aux_rows = [], []
    with torch.no_grad():
        for b in loader:
            ct, pet, mask = b['ct'].float().to(device), b['pet'].float().to(device), b['mask'].float().to(device)
            ids = b['idx'].cpu().numpy().tolist()
            bo, no = base(ct, pet, target_size=mask.shape[-2:]), new(ct, pet, target_size=mask.shape[-2:])
            blogit = bo['pred'] if isinstance(bo, dict) else bo[0]
            nlogit = no['pred'] if isinstance(no, dict) else no[0]
            bp, npb = torch.sigmoid(blogit).squeeze(1).cpu().numpy(), torch.sigmoid(nlogit).squeeze(1).cpu().numpy()
            gt, ctn, petn = mask.squeeze(1).cpu().numpy(), ct.squeeze(1).cpu().numpy(), pet.squeeze(1).cpu().numpy()
            ast = aux_stat(no)
            for i, sid in enumerate(ids):
                g = (gt[i] > 0.5).astype(np.uint8)
                bm, nm = metric((bp[i] > args.threshold).astype(np.uint8), g), metric((npb[i] > args.threshold).astype(np.uint8), g)
                ratio = div(nm['pred_area'], nm['gt_area'])
                fg, bg = g.astype(bool), ~g.astype(bool)
                row = {'sample_id': sid, 'base_dice': bm['dice'], 'new_dice': nm['dice'], 'delta_dice': nm['dice'] - bm['dice'], 'base_hd95': bm['hd95'], 'new_hd95': nm['hd95'], 'delta_hd95': nm['hd95'] - bm['hd95'], 'base_area': bm['pred_area'], 'new_area': nm['pred_area'], 'gt_area': nm['gt_area'], 'new_recall': nm['recall'], 'new_precision': nm['precision'], 'new_fp': nm['fp'], 'new_fn': nm['fn'], 'area_ratio': ratio, 'error_type': 'under_seg' if ratio < 0.7 else ('over_seg' if ratio > 1.3 else 'area_ok'), 'pet_contrast': float(petn[i][fg].mean() - petn[i][bg].mean()) if fg.any() else 0.0, 'ct_contrast': float(ctn[i][fg].mean() - ctn[i][bg].mean()) if fg.any() else 0.0}
                rows.append(row); aux_rows.append({'sample_id': sid, **ast})
    add_q(rows, 'delta_dice', 'regression_group', ['regressed', 'neutral', 'improved'])
    add_q(rows, 'gt_area', 'lesion_size_group', ['small', 'medium', 'large'])
    per = os.path.join(out_dir, 'per_sample_regression.csv')
    with open(per, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(sorted(rows, key=lambda x: x['delta_dice']))
    write_group(os.path.join(out_dir, 'regression_group_summary.csv'), rows, 'regression_group')
    write_group(os.path.join(out_dir, 'lesion_size_summary.csv'), rows, 'lesion_size_group')
    write_group(os.path.join(out_dir, 'error_type_summary.csv'), rows, 'error_type')
    with open(os.path.join(out_dir, 'aux_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        keys = sorted({k for r in aux_rows for k in r.keys()}); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(aux_rows)
    overall = {'n': len(rows), 'base_dice': mean(rows, 'base_dice'), 'new_dice': mean(rows, 'new_dice'), 'delta_dice': mean(rows, 'delta_dice'), 'base_hd95': mean(rows, 'base_hd95'), 'new_hd95': mean(rows, 'new_hd95'), 'delta_hd95': mean(rows, 'delta_hd95')}
    with open(os.path.join(out_dir, 'overall_summary.json'), 'w') as f: json.dump(overall, f, indent=2)
    print(json.dumps(overall, indent=2)); print('saved to:', out_dir)


if __name__ == '__main__':
    main()
