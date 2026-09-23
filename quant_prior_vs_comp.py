# -*- coding: utf-8 -*-
"""Quantify retrieved prior vs affine-compensated PET against real PET (Missing path)."""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def _fg_bg_masks(mask, hw):
    m = F.interpolate((mask > 0.5).float(), size=hw, mode='nearest')
    return m[:, 0] > 0.5, m[:, 0] <= 0.5


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--root', default='/root/autodl-tmp/data/PCLT20K')
    ap.add_argument('--nbatches', type=int, default=6)
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.checkpoint_dir, 'ckpt.best_joint.pth.tar'), map_location='cpu', weights_only=False)
    saved = dict(ck['config']); saved.pop('checkpoint_dir', None)
    saved['root'] = args.root; saved['ct_pretrained_path'] = None; saved['pet_pretrained_path'] = None
    cfg = SegMDTConfig(args=saved)
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task.device = torch.device('cpu'); task.model.to('cpu')
    task.model.load_state_dict(ck['model'], strict=True); task.model.eval()
    m = task.model
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    _, _, tl = get_pclt20k_loaders_cipa_aligned(cfg.root, cfg.image_size_2d, 4, 0, cfg.random_state,
        False, 'none', cfg.norm_mode, cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir)
    agg = {k: [] for k in ['cos_prior_real', 'cos_comp_real', 'fg_cos_prior', 'fg_cos_comp',
                            'prior_rms', 'comp_rms', 'real_rms', 'gamma_mean', 'gamma_std']}
    for bi, b in enumerate(tl):
        if bi >= args.nbatches: break
        ct, pet, mask = b['ct'].float(), b['pet'].float(), b['mask'].float()
        cf = m._encode_ct(ct); pr = m._encode_pet(pet)
        prior, _ = m.module1.retrieve_pet_prior(cf)
        comp, g, be = m.pet_affine(cf, prior)
        for s in range(4):
            P, C, R = prior[s], comp[s], pr[s]
            fg, bg = _fg_bg_masks(mask, P.shape[-2:])
            cp = F.cosine_similarity(P, R, dim=1)
            cc = F.cosine_similarity(C, R, dim=1)
            agg['cos_prior_real'].append(float(cp.mean()))
            agg['cos_comp_real'].append(float(cc.mean()))
            if fg.any():
                agg['fg_cos_prior'].append(float(cp[fg].mean()))
                agg['fg_cos_comp'].append(float(cc[fg].mean()))
            agg['prior_rms'].append(float(P.pow(2).mean().sqrt()))
            agg['comp_rms'].append(float(C.pow(2).mean().sqrt()))
            agg['real_rms'].append(float(R.pow(2).mean().sqrt()))
            agg['gamma_mean'].append(float(g[s].mean()))
            agg['gamma_std'].append(float(g[s].std()))
    print(f"{'metric':22s} {'mean':>10s}")
    for k, v in agg.items():
        if v: print(f'{k:22s} {np.mean(v):10.4f}')
    print('\nInterpretation:')
    print(' cos_prior_real = cosine(retrieved prior, real PET)  -- how good retrieval alone is')
    print(' cos_comp_real  = cosine(affine comp, real PET)      -- whether affine improves or hurts')
    print(' fg_cos_* = same but foreground-only (lesion region)')


if __name__ == '__main__':
    main()
