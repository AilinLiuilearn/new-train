"""Summarize paired raw/EMA evaluations, with patient-cluster bootstrap Dice deltas."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def dice(counts):
    tp,fp,fn=counts[...,0],counts[...,1],counts[...,2]
    denominator=2*tp+fp+fn
    return np.divide(2*tp,denominator,out=np.ones_like(denominator,dtype=float),where=denominator>0)


def paired_bootstrap(directory,report,seed=2023,repeats=5000):
    samples={tag:[json.loads(x) for x in (directory/(tag+'-samples.jsonl')).read_text().splitlines()] for tag in ('raw','ema')}
    raw,ema=samples['raw'],samples['ema']
    if [(r['image_id'],r['case_id']) for r in raw]!=[(r['image_id'],r['case_id']) for r in ema]:
        raise ValueError('Paired sample identity mismatch')
    patients=sorted({r['case_id'] for r in raw});index={p:i for i,p in enumerate(patients)}
    counts=np.zeros((2,2,len(patients),3),dtype=np.float64)
    for t,rows in enumerate((raw,ema)):
        for row in rows:
            for m,mode in enumerate(('full','missing')):
                counts[t,m,index[row['case_id']]] += [row[mode][k] for k in ('tp','fp','fn')]
    computed=dice(counts.sum(axis=2))
    for t,tag in enumerate(('raw','ema')):
        for m,mode in enumerate(('full','missing')):
            if not np.isclose(computed[t,m],report['results'][tag][mode]['dice'],atol=1e-8):
                raise AssertionError('Per-image counts disagree with baseline aggregate Dice')
    rng=np.random.RandomState(seed);dist=[]
    for _ in range(repeats):
        ids=rng.randint(0,len(patients),size=len(patients))
        d=dice(counts[:,:,ids,:].sum(axis=2));delta=d[1]-d[0]
        dist.append([delta[0],delta[1],float(delta.mean())])
    array=np.array(dist)
    patient_scores=dice(counts)
    patient_delta=patient_scores[1]-patient_scores[0]
    return {'patients':len(patients),'samples':len(raw),'seed':seed,'repeats':repeats,
            'unit':'patient/case_id; all slices retained per sampled patient',
            'ema_minus_raw_percentile_95ci':{name:np.quantile(array[:,i],[.025,.975]).tolist() for i,name in enumerate(('full','missing','joint'))},
            'patient_dice_changes':{mode:{'improved':int((patient_delta[m]>0).sum()),
                                              'worsened':int((patient_delta[m]<0).sum()),
                                              'unchanged':int((patient_delta[m]==0).sum()),
                                              'mean':float(patient_delta[m].mean()),'median':float(np.median(patient_delta[m]))}
                                    for m,mode in enumerate(('full','missing'))},
            'limit':'Conditional on these saved EMA-selected checkpoints and this reused pilot split; not training-seed uncertainty or independent confirmation.'}


def analyze(root):
    reports=json.loads((root/'comparison.json').read_text())
    directories=sorted(p for p in root.iterdir() if (p/'report.json').is_file())
    if len(reports)!=len(directories):raise ValueError('Incomplete comparison')
    lines=['# 同轮 raw / EMA 对照','',
           '同一checkpoint、相同4514张切片、FP32、512、batch16、同阈值和HD95定义；不更新参数、不重算BN统计。训练队列保持暂停。',
           '', '| checkpoint | epoch | 权重 | Full Dice | Missing Dice | Joint | Full HD95 | Missing HD95 |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    details=[]
    for directory,r in zip(directories,reports):
        b=paired_bootstrap(directory,r)
        (directory/'paired-bootstrap.json').write_text(json.dumps(b,indent=2)+'\n')
        name=Path(r['checkpoint']).parent.name
        for tag in ('raw','ema'):
            v=r['results'][tag]
            lines.append(f'| {name} | {r["epoch"]} | {tag} | {v["full"]["dice"]:.6f} | {v["missing"]["dice"]:.6f} | {v["joint_dice"]:.6f} | {v["full"]["hd95"]:.3f} | {v["missing"]["hd95"]:.3f} |')
        updates=r['ema_updates'];decay=min(r['ema_decay'],(1+updates)/(10+updates))
        details.extend(['',f'## {name}', '',f'EMA实际更新次数{updates}；warmup后的末次decay={decay:.8f}。若临时视此系数为常数，局部半衰期约{math.log(.5)/math.log(decay):.1f}步；实际系数随更新变化，这不是整个历史的固定窗口。', '',
                        '| 路径 | 权重 | 漏检阳性切片数 | 误报像素 | 漏检像素 | precision | sensitivity |',
                        '|---|---|---:|---:|---:|---:|---:|'])
        for mode in ('full','missing'):
            for tag in ('raw','ema'):
                v=r['results'][tag][mode]
                details.append(f'| {mode} | {tag} | {v["missed_positive_count"]} | {v["fp_pixels"]} | {v["fn_pixels"]} | {v["precision"]:.6f} | {v["sensitivity"]:.6f} |')
        details+=['',f'Joint EMA−raw={r["ema_minus_raw"]["joint_dice"]:+.6f}；患者配对cluster bootstrap 95%区间={b["ema_minus_raw_percentile_95ci"]["joint"]}。',
                  '区间只描述该checkpoint和该数据集上的患者抽样变动，不覆盖训练种子或EMA选模偏差。']
        details.append('患者聚合Dice变化（不同于主指标的全像素聚合）：'+json.dumps(b['patient_dice_changes'],ensure_ascii=False))
        saved=next(row for row in (json.loads(x) for x in (Path(r['checkpoint']).parent/'train.jsonl').read_text().splitlines()) if int(row['epoch'])==r['epoch'])
        diff=r['results']['ema']['joint_dice']-saved['joint_dice']
        details.append(f'重评EMA Joint与原日志差值={diff:+.12g}。')
    lines+=details+['', '## 解释边界','',
        'EMA是维护平均权重的独立副本，不参与loss和反向传播；当前实现不直接改变raw的AdamW更新。已用实际ModelEMA与两个相同小网络验证维护EMA前后的raw轨迹逐张量相同，但这不是独立的大模型重训练对照。',
        '当前两个best均按EMA验证指标选出，因此不能将这里的raw称为“关闭EMA实验自己的最佳模型”。完全关闭EMA还会改变选模和早停，需逐epoch同时评估raw/EMA或另做固定预算对照才能回答。',
        'EMA同时平均浮点BN缓冲，整数计数器直接复制。本对照测的是完整state，未隔离参数平均和BN统计平均各自的贡献。',
        'val/test复用test.txt，属于pilot；两个checkpoint使用相同患者集，不能当成两次独立测试或多种子证据。没有改EMA系数、训练损失、增强或模型结构。',
        '', '参考：[PyTorch AveragedModel文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)。具体行为以本仓库utils/ema.py为准。']
    (root/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(root/'analysis.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    analyze(p.parse_args().root)
