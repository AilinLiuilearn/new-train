"""Wait for the authorized BN run, compare selected epochs, freeze an engineering baseline.

No training, hyperparameter search, checkpoint copying, or source-file edits.
Joint remains the selection rule; ties retain GN. Pilot uncertainty is explicit.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


FIELDS=('epoch','weights','full_dice','missing_dice','joint_dice','full_hd95','missing_hd95',
        'full_precision','missing_precision','full_sensitivity','missing_sensitivity',
        'full_missed_positive_count','missing_missed_positive_count')
FIXED=('stage','joint_from_pretrained','root','train_split_file','val_split_file','test_split_file',
       'allow_val_equals_test','image_size_2d','batch_size','num_workers','aug_mode','norm_mode','epochs',
       'learning_rate','weight_decay','cosine_warmup','cosine_min_lr','lr_flat_ratio','mixed_precision',
       'ema_enabled','ema_decay','ema_start_epoch','grad_clip','random_state','hd95_mode')


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def best_row(run):
    rows=[json.loads(x) for x in (run/'train.jsonl').read_text().splitlines() if x.strip()]
    if not rows or any(not math.isfinite(r['joint_dice']) for r in rows):raise ValueError('Invalid Joint log')
    row=max(rows,key=lambda r:r['joint_dice'])
    if not math.isclose(row['joint_dice'],.5*(row['full_dice']+row['missing_dice']),abs_tol=1e-12):
        raise ValueError('Joint is not the agreed arithmetic mean')
    return {k:row[k] for k in FIELDS},len(rows)


def select(gn,bn):
    return 'BN' if bn['joint_dice']>gn['joint_dice'] else 'GN'


def finalize(gn_run,bn_run,review,output,allow_stopped=False):
    status=json.loads((review/'status.json').read_text())
    stopped=status['state']=='paused' and allow_stopped
    if status['state']!='complete' and not stopped:raise ValueError('BN run must finish successfully before freezing')
    final=None if stopped else json.loads((bn_run/'final_test.json').read_text())
    configs={tag:json.loads((run/'config.json').read_text()) for tag,run in [('GN',gn_run),('BN',bn_run)]}
    differences={key:[configs['GN'].get(key),configs['BN'].get(key)] for key in FIXED if configs['GN'].get(key)!=configs['BN'].get(key)}
    if differences:raise ValueError('Scientific config mismatch: '+json.dumps(differences))
    if configs['GN'].get('decoder_norm','GN')!='GN' or configs['BN'].get('decoder_norm')!='BN':raise ValueError('Unexpected decoder norms')
    if json.loads((gn_run/'splits.json').read_text())!=json.loads((bn_run/'splits.json').read_text()):raise ValueError('Split signatures differ')
    for tag,run in [('GN',gn_run),('BN',bn_run)]:
        cfg=configs[tag]
        if cfg.get('stage1_checkpoint') or cfg.get('crop_guidance','none')!='none' or cfg.get('copy_paste','none')!='none':raise ValueError('Unexpected task initialization or augmentation')
    initial={tag:json.loads((run/'initialization.json').read_text()) for tag,run in [('GN',gn_run),('BN',bn_run)]}
    for key in ('ct','pet'):
        if initial['GN'][key]['sha256']!=initial['BN'][key]['sha256']:raise ValueError('Generic initialization differs')
    gn,gn_epochs=best_row(gn_run);bn,bn_epochs=best_row(bn_run)
    if stopped:
        from run_staged import load
        checkpoint=load(bn_run/'best.pt')
        if (checkpoint['epoch']!=bn['epoch'] or checkpoint['config'].get('decoder_norm')!='BN'
            or not math.isclose(checkpoint['val']['joint_dice'],bn['joint_dice'],abs_tol=1e-9)):
            raise ValueError('Saved stopped-run BN checkpoint does not match selected validation epoch')
        del checkpoint
    elif final['epoch']!=bn['epoch'] or not math.isclose(final['metrics']['joint_dice'],bn['joint_dice'],abs_tol=1e-9):raise ValueError('Final BN checkpoint does not match selected epoch')
    selected=select(gn,bn);run=bn_run if selected=='BN' else gn_run
    comparison={'GN':gn,'BN':bn,'completed_epochs':{'GN':gn_epochs,'BN':bn_epochs},
        'BN_minus_GN':{k:bn[k]-gn[k] for k in FIELDS if k not in ('epoch','weights')},
        'selected_decoder_norm':selected,'rule':'Higher selected-checkpoint Joint; exact tie retains GN. Same-epoch Full/Missing per run.',
        'BN_run_status':'user_stopped_before_early_stopping' if stopped else 'complete',
        'BN_final_test_executed':not stopped,
        'decision_scope':'Operational choice from a single-seed pilot point estimate, not proof of statistical superiority.',
        'tradeoffs':[k for k in ('missing_dice','full_dice','full_sensitivity','missing_sensitivity','full_precision','missing_precision')
                     if (bn if selected=='BN' else gn)[k]<(gn if selected=='BN' else bn)[k]],
        'observed_policy':{'GN_saved_patience':configs['GN'].get('early_stopping_patience'),
                           'BN_saved_patience':configs['BN']['early_stopping_patience'],'future_patience':5},
        'limits':['Same development data; val=test.txt reused. No independent final test.',
                  'Two configurations are not a multi-seed uncertainty analysis.',
                  'GN reference was paused; observed completed budget is reported without pretending identical histories.',
                  'User-stopped BN results do not establish convergence or an exhausted patience window.' if stopped else 'BN completed its configured stopping rule.',
                  'Precision and sensitivity are reported alongside empty-prediction positive-slice counts; not 3D lesion detection.']}
    atomic(output/'comparison.json',comparison)
    cfg={k:v for k,v in configs[selected].items() if not k.startswith('_')}
    for key in ('output','resume','evaluate','guidance_identity','copy_paste_identity'):cfg.pop(key,None)
    cfg.update(decoder_norm=selected,early_stopping_patience=5,save_last_checkpoint=False,
               stage1_checkpoint=None,joint_from_pretrained=True,crop_guidance='none',copy_paste='none',
               guidance_map_dir=None,paste_index=None)
    atomic(output/'frozen_training_config.json',cfg)
    atomic(output/'reference.json',{'run':str(run.resolve()),'best_checkpoint':str((run/'best.pt').resolve()),
                                  'checkpoint_sha256':sha(run/'best.pt'),'future_initialization':'generic encoders, not this task checkpoint'})
    lines=['# 已冻结的模型与训练协议','',f'解码器选择：**{selected}**。按预先记录的Joint规则做本次工程选择，不声称单次pilot证明普遍最优。',
           'BN由用户要求提前停止，未达到5轮无提升早停；中断轮次未完成验证，不计入完成轮数。这里使用已保存最佳checkpoint的验证指标，未执行中断后的final_test。' if stopped else 'BN已正常完成配置的停止流程。',
           '', '| 模型 | epoch | Full Dice | Missing Dice | Joint | Full HD95 | Missing HD95 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for tag,row in [('GN',gn),('BN',bn)]:
        lines.append(f'| {tag} | {row["epoch"]} | {row["full_dice"]:.6f} | {row["missing_dice"]:.6f} | {row["joint_dice"]:.6f} | {row["full_hd95"]:.3f} | {row["missing_hd95"]:.3f} |')
    lines+=['', 'CT固定存在、PET整幅可缺失。CT ConvNeXtV2-Nano + PET MiT-B1，四尺度64/128/320/512，对齐BN，相加融合，两个独立'+selected+'解码器。',
        '每图同时Full/Missing，共享一次CT编码，相同GT，loss=0.5×CT_loss+0.5×Full_loss；Missing推理禁止PET/Full模块。',
        '通用编码器预训练初始化；512/batch16/lr8e-5；原AdamW、CIPA随机配对增强、归一化、AMP、EMA及60轮调度保持；5轮无严格Joint提升早停，仅best.pt。',
        '具体可复用参数见frozen_training_config.json。创新训练不得默认加载reference任务权重，不迁移其optimizer/EMA。',
        '', '完整敏感度、precision、漏检切片及指标取舍见comparison.json。selected模型较另一模型下降的指标：'+str(comparison['tradeoffs'])+'。',
        '原GN配置早停10；实际完成'+str(gn_epochs)+'轮。BN配置早停5、实际完成'+str(bn_epochs)+'轮，均保持60轮调度，不能掩盖历史差别。',
        '', 'val/test复用，单种子pilot，不是独立最终测试。差异小也按既定规则选一个固定工程基线，不宣称统计显著或全局最优。',
        '此后集中验证创新，一次改变一个因素；不因某个模块未涨分而临时改基础结构、EMA、损失或预算。若发现可复现实现错误，修复并记录受影响实验；正式研究验证与独立患者测试仍须完成。',
        '本流程只比较与归档，不自动启动新的训练或创新模块。']
    (output/'frozen_framework.md').write_text('\n'.join(lines)+'\n')
    return comparison


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gn_run',type=Path,default=Path('checkpoints_new/STAGED/dual-add-joint-cropcontrol-s2023-v1'))
    p.add_argument('--bn_run',type=Path,default=Path('checkpoints_new/STAGED/dual-add-joint-bndecoder-s2023-v1'))
    p.add_argument('--review',type=Path,default=Path('checkpoints_new/AUGMENTATION/bn-baseline-review-s2023-v1'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    state={'state':'waiting','pid':os.getpid(),'started_unix':time.time(),
           'selection_rule':'Higher selected-checkpoint Joint; tie GN; report all branch/HD95 tradeoffs.',
           'training_modified':False,'no_new_training':True}
    atomic(a.output/'status.json',state)
    try:
        while True:
            try:s=json.loads((a.review/'status.json').read_text())
            except json.JSONDecodeError:
                if not a.wait:raise
                time.sleep(1);continue
            if s['state']=='complete':break
            if s['state']!='running':raise RuntimeError('BN run is not running/complete: '+str(s['state']))
            if not a.wait:raise RuntimeError('BN run unfinished; use --wait')
            cmdline=Path('/proc')/str(s['pid'])/'cmdline'
            try:alive=bool(cmdline.read_bytes())
            except FileNotFoundError:alive=False
            if not alive:
                # The trainer can finish between reading status and checking /proc.
                latest=json.loads((a.review/'status.json').read_text())
                if latest['state']=='complete':break
                raise RuntimeError('Trainer exited without completed status; no baseline frozen')
            time.sleep(30)
        result=finalize(a.gn_run,a.bn_run,a.review,a.output)
        state.update(state='frozen',selected_decoder_norm=result['selected_decoder_norm'],ended_unix=time.time())
    except BaseException as error:
        state.update(state='failed',error=repr(error),ended_unix=time.time());raise
    finally:atomic(a.output/'status.json',state)
