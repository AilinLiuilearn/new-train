# -*- coding: utf-8 -*-
"""Render Missing-path stage maps with a PCA-1 channel projection (not mean).

Channel-mean hides structure because S4 channels carry anti-correlated
spatial patterns; PCA-1 exposes the dominant spatial component.
"""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def n01(x):
    x = x - x.min(); d = x.max() - x.min()
    return x / d if d > 1e-8 else x * 0


def pca1(feat):  # [C,h,w] -> [h,w] normalised
    C, h, w = feat.shape
    X = feat.reshape(C, -1).float(); X = X - X.mean(1, keepdim=True)
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    return n01((X.t() @ U[:, 0]).reshape(h, w).numpy())


def gem(feat):  # generalized-mean pool over channels [C,h,w]->[h,w]
    return n01(feat.abs().float().pow(2).mean(0).sqrt().numpy())


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--root', default='/root/autodl-tmp/data/PCLT20K')
    ap.add_argument('--ncases', type=int, default=6)
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.checkpoint_dir, 'ckpt.best_joint.pth.tar'), map_location='cpu', weights_only=False)
    s = dict(ck['config']); s.pop('checkpoint_dir', None)
    s['root'] = args.root; s['ct_pretrained_path'] = None; s['pet_pretrained_path'] = None
    cfg = SegMDTConfig(args=s)
    t = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    t.device = torch.device('cpu'); t.model.to('cpu')
    t.model.load_state_dict(ck['model'], strict=True); t.model.eval(); m = t.model
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    _, _, tl = get_pclt20k_loaders_cipa_aligned(cfg.root, cfg.image_size_2d, 4, 0, cfg.random_state,
        False, 'none', cfg.norm_mode, cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir)
    out = os.path.join(args.checkpoint_dir, 'viz3'); os.makedirs(out, exist_ok=True)
    done = 0
    for b in tl:
        ct, pet, mask = b['ct'].float(), b['pet'].float(), b['mask'].float()
        cf = m._encode_ct(ct); pr = m._encode_pet(pet)
        prior, _ = m.module1.retrieve_pet_prior(cf)
        comp, g, be = m.pet_affine(cf, prior)
        for i in range(ct.shape[0]):
            if done >= args.ncases: break
            r4, p4, c4 = pr[-1][i], prior[-1][i], comp[-1][i]   # [C,h,w]
            gt = mask[i, 0].numpy()
            fig, ax = plt.subplots(2, 5, figsize=(21, 8.5))
            rows = [
                (ct[i, 0].numpy(), 'CT', 'gray'),
                (pca1(r4), 'PET_real S4  [PCA1]', 'gray'),
                (pca1(p4), 'P_prior S4  [PCA1]', 'gray'),
                (pca1(c4), 'P_comp S4  [PCA1]', 'gray'),
                (n01((c4 - p4).abs().mean(0).numpy()), '|P_comp-P_prior| chmean', 'gray'),
                (gt, 'GT', 'gray'),
                (gem(r4), 'PET_real S4  [ch-RMS]', 'gray'),
                (gem(p4), 'P_prior S4  [ch-RMS]', 'gray'),
                (gem(c4), 'P_comp S4  [ch-RMS]', 'gray'),
                (n01((c4 - r4).abs().mean(0).numpy()), '|P_comp-PET_real| chmean', 'gray'),
            ]
            for a, (img, ti, cm) in zip(ax.ravel(), rows):
                a.imshow(img, cmap=cm); a.set_title(ti, fontsize=9); a.axis('off')
            fig.suptitle(f"case={b['case_id'][i]}  prior_rms={float(prior[-1][i].pow(2).mean().sqrt()):.2f} "
                         f"comp_rms={float(comp[-1][i].pow(2).mean().sqrt()):.2f} "
                         f"real_rms={float(pr[-1][i].pow(2).mean().sqrt()):.2f}", fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'proj_{done:02d}_{b["case_id"][i]}.png'), dpi=90)
            plt.close(fig); done += 1
        if done >= args.ncases: break
    print('saved', done, '->', out)


if __name__ == '__main__':
    main()
