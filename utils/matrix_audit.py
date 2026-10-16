"""Frozen diagnostic interventions. Full/GT oracles must never be deployment results."""
import torch
import torch.nn.functional as F


@torch.inference_mode()
def audit_scale(scale, x, full, mask, samples=128):
    banks = [b for b in scale.banks if bool(b.valid)]
    if not banks:
        return dict(normal=x, uniform=x, shuffle=x, g010=x, g030=x, g100=x), []
    keys = torch.cat([b.keys for b in banks]).float()
    matrices = torch.cat([b.transforms for b in banks]).float()
    q = x.float().flatten(2).transpose(1, 2)
    a = (F.normalize(q, dim=-1) @ keys.T / scale.temperature).softmax(-1)
    normal, uniform, shuffle = [torch.zeros_like(q) for _ in range(3)]
    for j, matrix in enumerate(matrices):
        value = q @ matrix.T
        normal += a[..., j:j+1]*value
        uniform += value/len(matrices)
        shuffle += a[..., (j+1)%len(matrices):(j+1)%len(matrices)+1]*value
    def image(v): return v.transpose(1,2).reshape_as(x)
    def gated(v):
        v = image(v)
        gate = scale.gate(torch.cat([x.float(), v], 1)).sigmoid()
        return (x+gate*(v-x)).to(x.dtype)
    result = dict(normal=gated(normal), uniform=gated(uniform), shuffle=gated(shuffle))
    for name, g in [('g010', .1), ('g030', .3), ('g100', 1.)]:
        result[name] = (x+g*(image(normal)-x)).to(x.dtype)
    occupancy = F.interpolate(mask.float(), size=x.shape[-2:], mode='area').flatten(1)
    target = full.float().flatten(2).transpose(1,2)
    stats = []
    for b in range(len(x)):
        for label in ['foreground','background']:
            weights = occupancy[b] if label=='foreground' else 1-occupancy[b]
            ids = (weights>0).nonzero(as_tuple=True)[0]
            if not len(ids): continue
            # Deterministic bounded uniform index sample; does not modify global RNG.
            ids = ids[torch.linspace(0,len(ids)-1,min(samples,len(ids)),device=x.device).long()]
            v, f, w = q[b,ids], target[b,ids], weights[ids]
            values = torch.stack([v@m.T for m in matrices], 1)
            errors = (values-f[:,None]).square().mean(-1)
            base = (v-f).square().mean(-1)
            transformed = (normal[b,ids]-f).square().mean(-1)
            att = a[b,ids]
            correction = (result['normal'][b].flatten(1).T[ids]-v).square().mean(-1)
            stats.append(dict(batch_index=b,region=label,sample_count=len(ids),mass=float(w.sum()),
                ct_error_sum=float((base*w).sum()),normal_error_sum=float((transformed*w).sum()),
                oracle_error_sum=float((errors.min(1).values*w).sum()),
                correction_energy_sum=float((correction*w).sum()),ct_energy_sum=float((v.square().mean(-1)*w).sum()),
                attention_entropy_sum=float((-(att*att.clamp_min(1e-12).log()).sum(-1)*w).sum()),
                attention_deviation_sum=float(((att-1/len(matrices)).abs().sum(-1)*w).sum()),
                value_variance_sum=float((values.var(1,unbiased=False).mean(-1)*w).sum())))
    return result, stats


def audit_findings(summary, feature_rows, tolerance=.001):
    """Heuristic flags, not causal proof. Tolerance is raw Dice (default .1 pp)."""
    d = {name: value['dice']-summary['off']['dice'] for name,value in summary.items()}
    findings = []
    if abs(d['normal']) <= tolerance:
        findings.append('正常补偿的总体 Dice 收益未超过诊断阈值；不等同统计显著性结论。')
    if abs(summary['normal']['dice']-summary['uniform']['dice']) <= tolerance and abs(summary['normal']['dice']-summary['shuffle']['dice']) <= tolerance:
        findings.append('正常检索未明显优于均匀／错配对照，当前内容检索的任务价值不足。')
    if max(d[n] for n in ['g010','g030','g100']) > d['normal']+tolerance:
        findings.append('固定采用比例优于学习门控，优先检查门控采用；测试集扫描结果不能作为调参后的最终成绩。')
    if max(d[f'scale{i}'] for i in range(1,5)) > d['normal']+tolerance:
        findings.append('存在单尺度优于全尺度的现象，优先检查尺度干扰。')
    if d['full_all'] > d['normal']+tolerance:
        findings.append('真实 Full 特征仍有改善空间，当前矩阵补偿没有恢复全部有用信息。')
    for row in feature_rows:
        if row['oracle_error_sum'] >= row['ct_error_sum']:
            findings.append(f"{row['scale']}/{row['region']}：现有单矩阵的 Full 特征 oracle 在采样点上也未降低误差；检查变换泛化。")
        elif row['normal_error_sum'] >= row['ct_error_sum']:
            findings.append(f"{row['scale']}/{row['region']}：oracle 单矩阵可降误差而正常软检索未降，检查矩阵选择或混合抵消。")
        elif abs(d['normal']) <= tolerance:
            findings.append(f"{row['scale']}/{row['region']}：正常检索可降低 Full 特征误差，但总分割收益接近零；结合单尺度和采用比例对照判断任务目标不一致或门控／尺度干扰。")
    if not findings: findings.append('未触发上述定位规则，需查看病例差异和特征误差，不能自动认定设计有效。')
    return findings
