"""Opt-in native-resolution GFRM diagnostics; no training-state mutation."""
import json
import os
import random

import numpy as np
import torch


def visualize_gfrm(model, loader, cfg, device, epoch, weights_tag):
    if not cfg.gfrm_enabled or cfg.fusion_vis_samples <= 0:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    py_state, np_state = random.getstate(), np.random.get_state()
    tr_state = torch.get_rng_state()
    cu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(m, m.training) for m in model.modules()]
    paths = []
    try:
        model.eval()
        batch = next(iter(loader))
        n = min(int(cfg.fusion_vis_samples), len(batch['ct']))
        with torch.no_grad():
            out = model(batch['ct'][:n].to(device), pet=batch['pet'][:n].to(device),
                        forward_mode='full', return_diagnostics=True, return_maps=True)
        info = out['fusion_diagnostics']
        folder = os.path.join(cfg.checkpoint_dir, 'fusion_visualizations', f'epoch_{epoch:03d}')
        os.makedirs(folder, exist_ok=True)
        summary = {'epoch': epoch, 'weights': weights_tag,
                   'map_definition': 'channel RMS of signed features; not lesion probabilities',
                   'stages': {s:{k:v for k,v in d.items() if k != 'maps'} for s,d in info.items()}}
        meta = os.path.join(folder, 'diagnostics.json')
        with open(meta, 'w') as f:
            json.dump(summary, f, indent=2)
        paths.append(meta)
        for index in range(n):
            arrays = {'ct_input': batch['ct'][index].cpu().numpy(),
                      'pet_input': batch['pet'][index].cpu().numpy(),
                      'gt': batch['mask'][index].cpu().numpy(),
                      'prob': out['logits'][index].sigmoid().float().cpu().numpy()}
            for scale, stats in info.items():
                for name, value in stats['maps'].items():
                    a = value[index].numpy()
                    # Native-resolution RMS maps, not gigabytes of channel tensors.
                    arrays[f'{scale}_{name}'] = (a if name == 'channel_weights'
                                                else np.sqrt(np.mean(a*a,axis=0)))
            base = os.path.join(folder, f'sample_{index:02d}')
            np.savez_compressed(base+'.npz', **arrays)
            paths.append(base+'.npz')
            fig, axes = plt.subplots(5, 6, figsize=(18, 13))
            for ax in axes.flat:
                ax.axis('off')
            def display_image(a):
                a = np.asarray(a)
                if a.ndim == 3:
                    a = a[0]
                return a
            for j, name in enumerate(('ct_input','pet_input','gt','prob')):
                a = display_image(arrays[name]).copy()
                if name.endswith('input') and cfg.norm_mode == 'imagenet':
                    a = np.clip(a * .229 + .485, 0, 1)
                axes[0,j].imshow(a, cmap='gray', vmin=0 if name in ('gt','prob') else None,
                                 vmax=1 if name in ('gt','prob') else None)
                axes[0,j].set_title(name)
            names = ('ct','pet','delta_ct','delta_pet','fused','add')
            for row in range(1,5):
                for col, name in enumerate(names):
                    heat = arrays[f's{row}_{name}']
                    axes[row,col].imshow(heat, cmap='magma', vmin=0,
                                         vmax=max(float(heat.max()), 1e-8))
                    axes[row,col].set_title(f'S{row} {name} RMS')
            fig.suptitle(f'GFRM epoch={epoch}, weights={weights_tag}; heatmaps scaled independently')
            fig.tight_layout()
            fig.savefig(base+'.png', dpi=110)
            plt.close(fig)
            paths.append(base+'.png')
        return paths
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(tr_state)
        if cu_state is not None:
            torch.cuda.set_rng_state_all(cu_state)
