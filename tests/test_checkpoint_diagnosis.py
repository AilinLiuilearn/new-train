"""Checks for paired attribution, legacy restoration and end-to-end outputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import diagnose_add_vs_cainet as d


def test_pixel_changes_and_lesion_coverage():
    gt = np.zeros((16, 16), bool)
    gt[2:4, 2:4] = True
    gt[10:12, 10:12] = True
    add = gt.copy(); add[2:4, 2:4] = False; add[0, 0] = True
    fusion = gt.copy(); fusion[10:12, 10:12] = False; fusion[15, 15] = True
    row, lesions = d.compare_masks(gt, add, fusion, add, gt.astype('uint8'),
                                   hd95_mode='surface')
    assert row['corrected_fn'] == row['introduced_fn'] == 4
    assert row['corrected_fp'] == row['introduced_fp'] == 1
    assert row['delta_dice'] == 0
    assert row['gt_components'] == 2
    assert [r['fusion_detected'] for r in lesions] == [1, 0]
    assert [r['add_detected'] for r in lesions] == [0, 1]


def test_aggregation_is_global_counts_not_mean_dice():
    rows = []
    for n, error in [(4, True), (64, False)]:
        gt = np.zeros((16, 16), bool); gt.flat[:n] = True
        pred = np.zeros_like(gt) if error else gt.copy()
        row, _ = d.compare_masks(gt, pred, pred, pred, gt, hd95_mode='surface')
        rows.append(row)
    result = d.aggregate(rows, 'add')
    assert result['dice'] == pytest.approx(128/132)
    assert result['mean_slice_dice'] == .5


def test_empty_gt_and_boundary_rules():
    empty = np.zeros((16, 16), bool)
    assert d.boundary_f1(empty, empty, 2) == 1
    positive = empty.copy(); positive[5, 5] = True
    row, lesions = d.compare_masks(empty, positive, empty, empty, empty,
                                   hd95_mode='surface')
    assert row['add_dice'] == 0 and row['fusion_dice'] == 1
    assert row['corrected_fp'] == 1 and not lesions
    # Official HD95 extension is explicit and does not alter Dice counts.
    row = d.describe_mask(positive, empty, empty, empty, empty, empty, 2, 'cipa_official')
    assert row['hd95_negative_gt_extension'] == 1
    assert row['fp'] == 1


def test_checkpoint_resolution_and_ema_selection(tmp_path):
    with pytest.raises(FileNotFoundError):
        d.resolve_checkpoint(tmp_path)
    best = tmp_path/'ckpt.best_full.pth.tar'; best.touch()
    assert d.resolve_checkpoint(tmp_path) == best
    raw, ema = {'x': torch.tensor(1)}, {'x': torch.tensor(2)}
    assert d.select_weights(dict(model=raw, model_ema=ema, eval_weights='raw'), 'auto')[0] is raw
    assert d.select_weights(dict(model=raw, model_ema=ema, eval_weights='ema'), 'auto')[0] is ema
    with pytest.raises(ValueError, match='Ambiguous'):
        d.select_weights(dict(model=raw, model_ema=ema), 'auto')
    with pytest.raises(ValueError, match='does not contain'):
        d.select_weights(dict(model=raw), 'ema')


def test_small_foreground_feature_statistics_survive_downsampling():
    c = torch.ones(1, 2, 2, 2); p = c.clone()
    target = torch.zeros(1, 1, 8, 8); target[:, :, 3, 3] = 1
    stats = d.feature_statistics([c], [p], [c+p+1], target, ['x'])[0]
    assert stats['fg_delta_relative_l2'] == pytest.approx(.5)
    assert stats['bg_delta_relative_l2'] == pytest.approx(.5)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.pet_align = torch.nn.Linear(1, 1, bias=False)
        self.fusion = ToyFusion()

    def _decode(self, features, target_size):
        return {'logits': features[0][:, :1]*self.weight}

    def forward(self, ct, pet, forward_mode='full'):
        return self._decode(self.fusion([ct]*4, [pet]*4), ct.shape[-2:])


class ToyFusion(torch.nn.Module):
    def forward(self, c, p):
        return [x+y for x, y in zip(c, p)]


def saved_config(root, kind):
    return dict(root=str(root), norm_mode='cipa', image_size_2d=16,
                ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                decoder_channels=[512, 256, 128, 64], train_split_file='train.txt',
                val_split_file='test.txt', test_split_file='test.txt',
                fusion_type='AddFusion' if kind == 'add' else 'CAINetFusion',
                experiment_type='full_add_baseline' if kind == 'add' else 'full_cainet_fusion',
                cacr_enabled=False, gcm_enabled=False)


def test_strict_legacy_loading_and_saved_raw_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(d, 'build_dual_model', lambda cfg: {'model': ToyModel()})
    model = ToyModel(); model.pet_align = torch.nn.Identity()
    path = tmp_path/'legacy.pt'
    torch.save(dict(config=saved_config(tmp_path, 'add'), model=model.state_dict(),
                    model_ema={k: v+10 for k, v in model.state_dict().items()}, eval_weights='raw'), path)
    loaded, _, metadata = d.load_model(path, 'add')
    assert isinstance(loaded.pet_align, torch.nn.Identity)
    assert loaded.weight == 1 and metadata['weights'] == 'raw'
    assert not metadata['pet_alignment']
    state = model.state_dict(); state['bad_key'] = torch.ones(1)
    torch.save(dict(config=saved_config(tmp_path, 'add'), model=state), path)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        d.load_model(path, 'add')


def test_end_to_end_synthetic_checkpoints_and_report(monkeypatch, tmp_path):
    monkeypatch.setattr(d, 'build_dual_model', lambda cfg: {'model': ToyModel()})
    monkeypatch.setattr(d, 'build_full_cainet_model', lambda cfg: {'model': ToyModel()})
    root = tmp_path/'data'; root.mkdir()
    (root/'test.txt').write_text('001_001\n002_001\n')
    (root/'train.txt').write_text('003_001\n')
    for case in ('001', '002'):
        folder = root/case; folder.mkdir()
        for suffix in ('CT', 'PET', 'mask'):
            image = np.zeros((16, 16), np.uint8)
            image[4:10, 4:10] = 255
            Image.fromarray(image).save(folder/f'{case}_001_{suffix}.png')
    for kind in ('add', 'fusion'):
        state = ToyModel().state_dict()
        if kind == 'fusion':
            state['weight'] = torch.tensor(-1.)
        torch.save(dict(config=saved_config(root, kind), model=state,
                        eval_weights='raw', epoch=3, best_epoch=3), tmp_path/f'{kind}.pt')
    output = tmp_path/'out'
    result = d.main(['--add_checkpoint', str(tmp_path/'add.pt'),
                     '--fusion_checkpoint', str(tmp_path/'fusion.pt'),
                     '--output_dir', str(output), '--device', 'cpu',
                     '--num_workers', '0', '--hd95_mode', 'surface', '--top_k', '1',
                     '--save_predictions'])
    assert result['models']['add']['dice'] == 1
    assert result['models']['fusion']['dice'] == 0
    assert result['paired']['harmed_slices'] == 2
    assert (output/'report.md').is_file()
    assert len(list((output/'top_harmed').glob('*.png'))) == 1
    assert len(list((output/'predictions').glob('*.npz'))) == 2
    assert json.loads((output/'summary.json').read_text())['metadata']['samples'] == 2
    for name in ('per_slice.csv', 'per_case.csv', 'per_lesion.csv', 'feature_scales.csv'):
        assert (output/name).stat().st_size > 0
    with pytest.raises(FileExistsError):
        d.main(['--output_dir', str(output)])
