"""Training-only Add reference and class-balanced relative BCE protection.

No GT or reference model is used by the fusion forward/inference. This is
our adaptation, NOT PNDC's native Reverse Audit/Forward Checksum algorithm.
"""
import copy
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from models.components.add_fusion import AddFusion
from tasks.mdt_seg import MDTSegTeacher


def relative_protection_loss(logits, reference_logits, target):
    if (logits.shape != reference_logits.shape or logits.shape != target.shape
            or logits.ndim != 4 or logits.shape[1] != 1
            or logits.device != reference_logits.device or logits.device != target.device):
        raise ValueError('Protection requires matching B,1,H,W logits/reference/binary target on one device')
    if (not torch.isfinite(logits).all() or not torch.isfinite(reference_logits).all()
            or not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all()):
        raise ValueError('Protection requires finite logits and binary GT')
    with torch.autocast(device_type=logits.device.type,enabled=False):
        ref = reference_logits.detach().float()
        labels = target.detach().float()
        correct = (ref >= 0).eq(labels.bool())
        excess = (F.binary_cross_entropy_with_logits(logits.float(),labels,reduction='none')
                  - F.binary_cross_entropy_with_logits(ref,labels,reduction='none')).relu()
        terms,stats = [],{}
        for cls,name in ((1,'fg'),(0,'bg')):
            selected = correct & (labels == cls)
            count = selected.sum()
            stats[f'protect_correct_{name}'] = count.detach()
            # Empty classes contribute zero and do not halve a nonempty class.
            if bool(count):
                terms.append((excess*selected).sum()/count)
        loss = torch.stack(terms).mean() if terms else logits.float().sum()*0
    return loss,stats


def initialize_from_add(student,cfg):
    """Strict same-recipe Add warm-start; optional frozen reference OUTSIDE student.

    The run starts a fresh optimizer/scheduler/EMA, not an interrupted resume.
    SHA/weight choice/epoch are stored in config/checkpoints for traceability.
    """
    path = getattr(cfg,'baseline_checkpoint',None)
    if not path:
        if cfg.protect_enabled:
            raise ValueError('protect_enabled requires --baseline_checkpoint')
        cfg.initialization_mode = 'from_scratch_or_encoder_pretrained'
        cfg.baseline_sha256 = cfg.baseline_weights_resolved = None
        return None
    checkpoint = MDTSegTeacher.load_state_dicts(path)
    saved = checkpoint.get('config',{})
    if (saved.get('model_arch') != 'dual_shared_add_baseline'
            or saved.get('experiment_type') != 'full_add_baseline'
            or saved.get('fusion_type') != 'AddFusion'
            or saved.get('use_deep_supervision',False) or saved.get('deep_supervision',False)):
        raise ValueError('Reference must be an original Full Add baseline checkpoint')
    for key in ('image_size_2d','norm_mode','ct_backbone','pet_backbone','decoder_channels'):
        a,b=saved.get(key),getattr(cfg,key)
        if key=='decoder_channels': a,b=tuple(a or ()),tuple(b)
        if a != b:
            raise ValueError(f'Baseline recipe mismatch {key}: checkpoint={a}, current={b}')
    choice = cfg.baseline_weights
    if choice == 'auto': choice=checkpoint.get('eval_weights','raw')
    if choice not in ('raw','ema'):
        raise ValueError(f'Unknown baseline weights tag: {choice}')
    state = checkpoint.get('model_ema' if choice=='ema' else 'model')
    current = student.state_dict()
    expected = {key for key in current if not key.startswith('fusion.')}
    if (not isinstance(state,dict) or set(state) != expected
            or any(not torch.is_tensor(state[k]) or current[k].shape != state[k].shape for k in expected)):
        raise ValueError('Baseline state keys/shapes do not exactly match shared student architecture')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Non-finite baseline state')
    current.update(state)
    student.load_state_dict(current,strict=True)
    cfg.baseline_weights_resolved=choice
    cfg.baseline_epoch=checkpoint.get('epoch')
    cfg.baseline_source_learning_rate=saved.get('learning_rate')
    cfg.baseline_source_ema_decay=saved.get('ema_decay')
    cfg.initialization_mode='warm_start_full_add_fresh_optimizer'
    hasher=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''): hasher.update(chunk)
    cfg.baseline_sha256=hasher.hexdigest()
    if not cfg.protect_enabled: return None
    reference=copy.deepcopy(student)
    reference.fusion=AddFusion()
    reference.load_state_dict(state,strict=True)
    reference.eval().requires_grad_(False)
    return reference
