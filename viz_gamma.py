# -*- coding: utf-8 -*-
"""Extra diagnostics: gamma/beta heatmaps + a compact PET_real|prior|comp montage."""
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


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--root', default='/root/autodl-tmp/data/PCLT20K')
    ap.add_argument('--ncases', type=int, default=4)
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
    out = os.path.join(args.checkpoint_dir, 'viz2'); os.makedirs(out, exist_ok=True)
    done = 0
    for b in tl:
        ct, pet, mask = b['ct'].float(), b['pet'].float(), b['mask'].float()
        cf = m._encode_ct(ct); pr = m._encode_pet(pet)
        prior, _ = m.module1.retrieve_pet_prior(cf)
        comp, g, be = m.pet_affine(cf, prior)
        for i in range(ct.shape[0]):
            if done >= args.ncases: break
            real4 = pr[-1][i].mean(0).numpy()
            pri4 = prior[-1][i].mean(0).numpy()
            com4 = comp[-1][i].mean(0).numpy()
            gam = g[-1][i].mean(0).numpy()
            bet = be[-1][i].mean(0).numpy()
            gt = mask[i, 0].numpy()
            fig, ax = plt.subplots(2, 5, figsize=(20, 8))
            for a, (img, t, cm) in zip(ax.ravel(), [
                (ct[i, 0].numpy(), 'CT', 'gray'),
                (real4, 'PET_real S4', 'gray'),
                (pri4, 'P_prior S4', 'gray'),
                (com4, 'P_comp S4', 'gray'),
                (n01(com4 - pri4), '|P_comp-P_prior|', 'gray'),
                (gt, 'GT', 'gray'),
                (gam, 'gamma_s4 (heat)', 'coolwarm'),
                (n01(bet), 'beta_s4 |.| (heat)', 'coolwarm'),
                (n01(com4 - real4), '|P_comp-PET_real|', 'gray'),
                (n01(pri4 - real4), '|P_prior-PET_real|', 'gray'),
            ]):
                a.imshow(img, cmap=cm); a.set_title(t, fontsize=9); a.axis('off')
            fig.suptitle(f"case={b['case_id'][i]}  gamma_s4[min={gam.min():.2f},max={gam.max():.2f},mean={gam.mean():.2f}]  "
                         f"prior_rms={float(prior[-1][i].pow(2).mean().sqrt()):.2f} comp_rms={float(comp[-1][i].pow(2).mean().sqrt()):.2f} "
                         f"real_rms={float(pr[-1][i].pow(2).mean().sqrt()):.2f}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'diag_{done:02d}_{b["case_id"][i]}.png'), dpi=90)
            plt.close(fig); done += 1
        if done >= args.ncases: break
    print('saved', done, '->', out)


if __name__ == '__main__':
    main()
