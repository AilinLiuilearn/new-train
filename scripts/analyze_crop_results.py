"""Analyze completed v1 runs and train-only map region distribution for v2."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from datasets.pclt20k_seg import _imread_grayscale, _resize_gray
from utils.dual_state_maps import MapCache, atomic_json, train_records
from utils.dual_state_augmentation import components
from scripts.run_crop_pilots import comparison


def analyze(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    state = json.loads(Path(args.suite_status).read_text())
    result = comparison(args.map_dir, state['runs'])
    manifest = json.loads((Path(args.map_dir)/'manifest.json').read_text())
    cache = MapCache(args.map_dir, manifest['baseline_config'])
    records = train_records(manifest['baseline_config'], manifest['train_split_sha256'])
    indices = np.random.RandomState(args.seed).permutation(len(records))[:args.num_samples]
    areas_all = []
    expectations = {key: {'samples': 0, 'expected_area_sum': 0., 'background_probability_sum': 0.,
                         'tiny_probability_sum': 0.} for key in ('uniform', 'sqrt_area')}
    for step, idx in enumerate(indices, 1):
        record = records[idx]
        d = cache.read(record['image_id'])
        gt = _imread_grayscale(record['mask_path']) > 0
        if gt.shape != d.shape:
            gt = _resize_gray(gt.astype(np.uint8), d.shape[0], nearest=True).astype(bool)
        available = []
        for category in (1, 2):
            n, labels, stats, _ = components(d == category)
            if n <= 1: continue
            areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
            touching = np.bincount(labels[gt], minlength=n)[1:] > 0
            areas_all.extend(areas.tolist())
            available.append((areas, touching))
        if available:
            for policy in expectations:
                e = expectations[policy]
                e['samples'] += 1
                for areas, touching in available:
                    weights = np.ones_like(areas) if policy == 'uniform' else np.sqrt(areas)
                    weights /= weights.sum()*len(available)  # retain equal class chance
                    e['expected_area_sum'] += float((weights*areas).sum())
                    e['background_probability_sum'] += float(weights[~touching].sum())
                    e['tiny_probability_sum'] += float(weights[areas <= 2].sum())
        if step == 1 or step % 100 == 0 or step == len(indices):
            print(f'[ANALYSIS] train maps {step}/{len(indices)}', flush=True)
    areas = np.asarray(areas_all)
    region_stats = {'sample_count': len(indices), 'seed': args.seed, 'region_count': len(areas),
                    'median_area': float(np.median(areas)),
                    'one_pixel_fraction': float((areas == 1).mean()),
                    'at_most_two_pixels_fraction': float((areas <= 2).mean()),
                    'area_quantiles': dict(zip(['p50','p75','p90','p99'], np.quantile(areas, [.5,.75,.9,.99]).tolist())),
                    'image_ids': [records[i]['image_id'] for i in indices],
                    'policy_expectations_before_geometry': {}}
    for policy, e in expectations.items():
        region_stats['policy_expectations_before_geometry'][policy] = {
            'mean_selected_area': e['expected_area_sum']/max(e['samples'], 1),
            'pure_background_probability': e['background_probability_sum']/max(e['samples'], 1),
            'at_most_two_pixels_probability': e['tiny_probability_sum']/max(e['samples'], 1)}
    result['train_region_analysis'] = region_stats
    result['training_augmentation'] = {}
    for mode, directory in state['runs'].items():
        path = Path(directory)/'augmentation.jsonl'
        if not path.is_file(): continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        totals = {key: sum(row.get(key, 0.) for row in rows)
                  for key in ('guided_attempt','guided_success','background_target','target_area_sum',
                              'target_foreground_sum','guided_window_iou_sum')}
        result['training_augmentation'][mode] = {
            'epochs': len(rows), **totals,
            'background_target_fraction': totals['background_target']/max(totals['guided_success'],1),
            'mean_selected_area': totals['target_area_sum']/max(totals['guided_success'],1)}
    result['decision'] = {
        'v1_conclusion': 'A3 has no observed advantage over A2; no evidence for the dual-state mechanism yet.',
        'hypothesis': 'Uniform region sampling overweights fragmented tiny training errors; test sublinear area sampling.',
        'single_change': 'region probability proportional to sqrt(area), within the same class; keep class balance and all other settings.',
        'controls': 'A3-v2 versus A3-v1 and A2-v2 versus A2-v1; A3-v2 versus A2-v2.',
        'limits': 'Training-map diagnosis motivates one algorithmic candidate. No validation/test hyperparameter sweep; exponent fixed at 0.5, not selected on held-out Dice. Not a proven causal explanation.'}
    atomic_json(output/'analysis.json', result)
    lines = ['# v1结果与v2单变量假设', '',
             '|策略|已记录轮数|选中轮|Full Dice|Missing Dice|Joint Dice|已完成|',
             '|---|---:|---:|---:|---:|---:|---|']
    for r in result['runs']:
        s=r['selected']
        lines.append(f'|{r["strategy"]}|{r["observed_epochs"]}|{s["epoch"]}|{s["full_dice"]:.6f}|{s["missing_dice"]:.6f}|{s["joint_dice"]:.6f}|{r["completed"]}|')
    lines += ['', '区域分析只使用训练GT/地图；验证集仍复用test，指标为开发结果。', '',
              f'固定随机{len(indices)}张训练图，共{len(areas)}个差异连通区域；中位面积{np.median(areas):.1f}像素，单像素比例{(areas==1).mean():.2%}，≤2像素比例{(areas<=2).mean():.2%}。', '',
              'v2只把类内区域均匀采样改为面积平方根加权。类别1/2各半、引导概率0.5、原裁剪范围、融合、损失、lr、种子、初始化和60轮上限/patience10全部保留。', '',
              '这是一项待证伪的训练采样假设，不能断言碎小错误造成了下降。A1也未提升，背景选区不是已证实的唯一原因。A3-v1与A2-v1相差仅0.00023，单种子不足以判断显著差异。', '',
              '训练地图的全量错误率很低，可能已过拟合；本轮不更换teacher、不在线建图、不调参搜索。']
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'region_stats': {k:v for k,v in region_stats.items() if k!='image_ids'},
                      'decision': result['decision']}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map_dir', required=True)
    p.add_argument('--suite_status', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--num_samples', type=int, default=1024)
    p.add_argument('--seed', type=int, default=2023)
    analyze(p.parse_args())
